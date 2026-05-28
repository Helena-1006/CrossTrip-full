import os
from collections import defaultdict

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

import metrics
from spot_utils import save_model


class Trainer:
    def __init__(self, model, args, logger):
        self.model = model
        self.args = args
        self.logger = logger
        self.optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc
        )
        self.device = args.device

    def unpack_batch(self, batch_data):
        (
            uid,
            pad_ori_ck,
            pad_dst_ck,
            pad_masked_dst_ck,
            pad_o_hour,
            pad_d_hour,
            pad_masked_d_hour,
            pad_ori_t,
            pad_dst_t,
            pad_ori_l,
            pad_dst_l,
            ori_pad,
            dst_pad,
            ori_rg,
            dst_rg,
            pad_ori_tag,
            pad_dst_tag,
            query_start_poi,
            query_start_hour,
            query_end_poi,
            query_end_hour,
            query_len,
            user_profile,
            query_vec,
            home_prompt_text,
        ) = batch_data

        return {
            'uid': uid.to(self.device),
            'ori_ck': pad_ori_ck.to(self.device),
            'dst_ck': pad_dst_ck.to(self.device),
            'masked_dst_ck': pad_masked_dst_ck.to(self.device),
            'o_hour': pad_o_hour.to(self.device),
            'd_hour': pad_d_hour.to(self.device),
            'masked_d_h': pad_masked_d_hour.to(self.device),
            'ori_t': pad_ori_t.to(self.device),
            'dst_t': pad_dst_t.to(self.device),
            'ori_l': pad_ori_l.to(self.device),
            'dst_l': pad_dst_l.to(self.device),
            'ori_pad': ori_pad.to(self.device),
            'dst_pad': dst_pad.to(self.device),
            'ori_rg': ori_rg.to(self.device),
            'dst_rg': dst_rg.to(self.device),
            'ori_tag': pad_ori_tag.to(self.device),
            'dst_tag': pad_dst_tag.to(self.device),
            'query_start_poi': query_start_poi.to(self.device),
            'query_start_hour': query_start_hour.to(self.device),
            'query_end_poi': query_end_poi.to(self.device),
            'query_end_hour': query_end_hour.to(self.device),
            'query_len': query_len.to(self.device),
            'user_profile': user_profile.to(self.device),
            'query_vec': query_vec.to(self.device),
            'home_prompt_text': home_prompt_text,
        }

    def train_epoch(self, train_loader, epoch_idx):
        self.model.train()
        loss_sum = 0.0
        align_sum, dec_sum, sem_sum, gen_sum, pair_sum, trans_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        sem_mix_sum = 0.0
        valid_step_cnt = 0
        bad_step_cnt = 0

        iter_wrapper = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_idx} Train")
        for step_idx, batch_data in iter_wrapper:
            batch = self.unpack_batch(batch_data)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(batch)
            loss = out['loss']

            if not torch.isfinite(loss):
                bad_step_cnt += 1
                if bad_step_cnt <= 3:
                    comp_msg = []
                    for key in ['align', 'decouple', 'semantic', 'gen', 'pair', 'transition', 'sem_mix']:
                        val = out.get(key, None)
                        if torch.is_tensor(val):
                            comp_msg.append(f"{key}={float(val.detach().float().mean().item()):.6f}")
                    self.logger.log(
                        f"[WARN] Non-finite loss at epoch={epoch_idx}, step={step_idx}. "
                        f"Skip this batch. Components: {' | '.join(comp_msg)}"
                    )
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            loss_sum += float(loss.item())
            align_sum += float(out['align'].item())
            dec_sum += float(out['decouple'].item())
            sem_sum += float(out['semantic'].item())
            gen_sum += float(out['gen'].item())
            pair_sum += float(out.get('pair', torch.tensor(0.0)).item())
            trans_sum += float(out.get('transition', torch.tensor(0.0)).item())
            sem_mix_sum += float(out.get('sem_mix', torch.tensor(0.0)).item())
            valid_step_cnt += 1

        self.scheduler.step()
        n = max(valid_step_cnt, 1)
        self.logger.log(
            f"Epoch {epoch_idx}/{self.args.epoch} | "
            f"Loss {loss_sum / n:.6f} | Align {align_sum / n:.6f} | "
            f"Dec {dec_sum / n:.6f} | Sem {sem_sum / n:.6f} | Gen {gen_sum / n:.6f} | Pair {pair_sum / n:.6f} | Trans {trans_sum / n:.6f} | "
            f"SemMix {sem_mix_sum / n:.4f} | ValidSteps {valid_step_cnt}/{len(train_loader)} | "
            f"Skipped {bad_step_cnt}"
        )
        return loss_sum / n

    def validate(self, valid_loader, epoch_idx, split_name="VAL"):
        self.model.eval()
        metrics_list = []

        with torch.no_grad():
            for batch_data in tqdm(valid_loader, desc=f"Epoch {epoch_idx} {split_name}"):
                batch = self.unpack_batch(batch_data)
                target_batch = batch['dst_ck']
                pred_batch = self.model.predict(batch)
                m = metrics.evaluate_sequences(target_batch, pred_batch)
                metrics_list.append(m)

        if len(metrics_list) == 0:
            ret = {'f1': 0.0, 'pairs_f1': 0.0, 'full_f1': 0.0, 'full_pairs_f1': 0.0, 'rep': 0.0, 'full_rep': 0.0}
        else:
            ret = {
                k: float(np.mean([m[k] for m in metrics_list]))
                for k in ['f1', 'pairs_f1', 'full_f1', 'full_pairs_f1', 'rep', 'full_rep']
            }

        self.logger.log(
            f"[{split_name}] Epoch {epoch_idx} "
            f"F1: {ret['f1']:.4f} | Pairs_F1: {ret['pairs_f1']:.4f} | "
            f"Full_F1: {ret['full_f1']:.4f} | Full_Pairs_F1: {ret['full_pairs_f1']:.4f} | "
            f"REP: {ret['rep']:.4f} | Full_REP: {ret['full_rep']:.4f}"
        )
        self.model.train()
        return ret

    def train(self, train_loader, valid_loader):
        stopping_dict = defaultdict(float)
        stopping_dict['best_f1'] = -1.0
        stopping_dict['best_pairs_f1'] = -1.0
        stopping_dict['best_full_f1'] = -1.0
        stopping_dict['best_full_pairs_f1'] = -1.0
        stopping_dict['best_full_combo'] = -1.0
        stopping_dict['best_combo'] = -1.0
        stopping_dict['best_primary'] = -1.0
        stopping_dict['early_stop_cnt'] = 0
        best_epoch = -1

        early_stop_metric = str(getattr(self.args, 'early_stop_metric', 'combo'))
        combo_beta = float(getattr(self.args, 'combo_beta', 4.0))
        use_f1_floor_filter = bool(getattr(self.args, 'use_f1_floor_filter', 1))
        f1_floor_margin = float(getattr(self.args, 'f1_floor_margin', 0.002))
        save_dual_best = bool(getattr(self.args, 'save_dual_best', 1))
        disable_checkpoint_save = bool(getattr(self.args, 'disable_checkpoint_save', 0))

        for e in range(self.args.epoch):
            self.train_epoch(train_loader, e)

            # When best_save is enabled, skip periodic epoch checkpoints and keep only best-model checkpoints.
            if (not disable_checkpoint_save) and (not bool(getattr(self.args, 'best_save', False))) and (e % self.args.save_step == 0):
                save_model(
                    self.model,
                    e,
                    self.args.save_path,
                    self.optimizer,
                    self.scheduler,
                    save_trainable_only=bool(getattr(self.args, 'save_trainable_only', 1)),
                    save_optimizer_state=bool(getattr(self.args, 'save_optimizer_state', 0)),
                )

            val_metrics = self.validate(valid_loader, e, split_name="VAL")
            current_f1 = val_metrics['f1']
            current_pairs = val_metrics['pairs_f1']
            current_full_f1 = val_metrics['full_f1']
            current_full_pairs = val_metrics['full_pairs_f1']
            current_combo = current_f1 + combo_beta * current_pairs
            current_full_combo = current_full_f1 + combo_beta * current_full_pairs

            if (not disable_checkpoint_save) and save_dual_best and current_f1 > stopping_dict['best_f1']:
                stopping_dict['best_f1'] = current_f1
                save_model(
                    self.model,
                    "best_f1",
                    self.args.save_path,
                    self.optimizer,
                    self.scheduler,
                    save_trainable_only=bool(getattr(self.args, 'save_trainable_only', 1)),
                    save_optimizer_state=bool(getattr(self.args, 'save_optimizer_state', 0)),
                )

            if (not disable_checkpoint_save) and save_dual_best and current_pairs > stopping_dict['best_pairs_f1']:
                stopping_dict['best_pairs_f1'] = current_pairs
                save_model(
                    self.model,
                    "best_pairs",
                    self.args.save_path,
                    self.optimizer,
                    self.scheduler,
                    save_trainable_only=bool(getattr(self.args, 'save_trainable_only', 1)),
                    save_optimizer_state=bool(getattr(self.args, 'save_optimizer_state', 0)),
                )

            if current_combo > stopping_dict['best_combo']:
                stopping_dict['best_combo'] = current_combo

            if current_full_f1 > stopping_dict['best_full_f1']:
                stopping_dict['best_full_f1'] = current_full_f1

            if current_full_pairs > stopping_dict['best_full_pairs_f1']:
                stopping_dict['best_full_pairs_f1'] = current_full_pairs

            if current_full_combo > stopping_dict['best_full_combo']:
                stopping_dict['best_full_combo'] = current_full_combo

            f1_floor = stopping_dict['best_f1'] - f1_floor_margin
            if stopping_dict['best_f1'] < 0.0:
                f1_floor = -1e9
            pass_floor = (not use_f1_floor_filter) or (current_f1 >= f1_floor)

            if early_stop_metric == 'f1':
                primary_score = current_f1
            elif early_stop_metric == 'pairs_f1':
                primary_score = current_pairs
            elif early_stop_metric == 'full_f1':
                primary_score = current_full_f1
            elif early_stop_metric == 'full_pairs_f1':
                primary_score = current_full_pairs
            elif early_stop_metric == 'full_combo':
                primary_score = current_full_combo
            else:
                primary_score = current_combo

            self.logger.log(
                f"[MODEL_SELECT] epoch={e} | metric={early_stop_metric} | "
                f"F1={current_f1:.4f} | Pairs={current_pairs:.4f} | "
                f"Full_F1={current_full_f1:.4f} | Full_Pairs={current_full_pairs:.4f} | "
                f"Combo={current_combo:.4f} | Full_Combo={current_full_combo:.4f} (beta={combo_beta:.2f}) | "
                f"F1_floor={f1_floor:.4f} | pass_floor={pass_floor}"
            )

            if pass_floor and primary_score > stopping_dict['best_primary']:
                stopping_dict['best_primary'] = primary_score
                stopping_dict['early_stop_cnt'] = 0
                stopping_dict['best_epoch'] = e
                best_epoch = e
                if not save_dual_best and current_f1 > stopping_dict['best_f1']:
                    stopping_dict['best_f1'] = current_f1
                if not save_dual_best and current_pairs > stopping_dict['best_pairs_f1']:
                    stopping_dict['best_pairs_f1'] = current_pairs
                if not disable_checkpoint_save:
                    save_model(
                        self.model,
                        "best",
                        self.args.save_path,
                        self.optimizer,
                        self.scheduler,
                        save_trainable_only=bool(getattr(self.args, 'save_trainable_only', 1)),
                        save_optimizer_state=bool(getattr(self.args, 'save_optimizer_state', 0)),
                    )
            else:
                stopping_dict['early_stop_cnt'] += 1

            if stopping_dict['early_stop_cnt'] >= self.args.stop_epoch:
                self.logger.log("Early stopped!")
                break

        return best_epoch


def load_checkpoint(model, ckpt_path, map_location='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint['state_dict']
    state_type = checkpoint.get('state_dict_type', 'full')
    strict = state_type == 'full'
    load_res = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing = getattr(load_res, 'missing_keys', [])
        unexpected = getattr(load_res, 'unexpected_keys', [])
        if len(unexpected) > 0:
            print(f"[WARN] Unexpected keys while loading trainable-only checkpoint: {unexpected[:8]}")
        if len(missing) > 0:
            print(f"[INFO] Missing keys expected for trainable-only checkpoint: {len(missing)} keys")
    return model
