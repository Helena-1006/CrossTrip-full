import os
import random
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _build_trainable_state_dict(model):
    named_params = dict(model.named_parameters())
    state = {}
    for k, v in model.state_dict().items():
        p = named_params.get(k, None)
        if p is not None and p.requires_grad:
            state[k] = v.detach().cpu()
    return state


def save_model(
    model,
    i,
    save_dir,
    optimizer=None,
    scheduler=None,
    save_trainable_only=False,
    save_optimizer_state=False,
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'model_{i}.xhr')

    if save_trainable_only:
        state_dict = _build_trainable_state_dict(model)
        payload = {
            "state_dict": state_dict,
            "state_dict_type": "trainable_only",
        }
    else:
        payload = {
            "state_dict": model.state_dict(),
            "state_dict_type": "full",
        }

    if save_optimizer_state and optimizer is not None and scheduler is not None:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()

    try:
        torch.save(payload, save_path)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint save failed at {save_path}. "
            f"Try enabling lightweight checkpoints (save_trainable_only=1) or free disk space. "
            f"Original error: {e}"
        )


def path_exist(path):
    os.makedirs(path, exist_ok=True)


class Logger(object):
    def __init__(self, log_path, name, seed, is_write_file=True):
        cur_time = time.strftime("%m-%d-%H:%M", time.localtime())
        self.is_write_file = is_write_file
        self.log_file = None
        if self.is_write_file:
            os.makedirs(log_path, exist_ok=True)
            self.log_file = open(os.path.join(log_path, "%s %s(%d).log" % (cur_time, name, seed)), 'w')

    def log(self, log_str):
        out_str = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {log_str}"
        print(out_str)
        if self.is_write_file and self.log_file is not None:
            self.log_file.write(out_str + '\n')
            self.log_file.flush()

    def close_log(self):
        if self.is_write_file and self.log_file is not None:
            self.log_file.close()


def collate_fn(batch):
    (
        uid,
        ori_ck,
        dst_ck,
        masked_dst_ck,
        o_hour,
        d_hour,
        masked_d_h,
        ori_t,
        dst_t,
        ori_l,
        dst_l,
        ori_rg,
        dst_rg,
        ori_tag,
        dst_tag,
        query_start_poi,
        query_start_hour,
        query_end_poi,
        query_end_hour,
        query_len,
        user_profile,
        query_vec,
        home_prompt_text,
    ) = zip(*batch)

    pad_ori_ck = pad_sequence(ori_ck, batch_first=True)
    pad_dst_ck = pad_sequence(dst_ck, batch_first=True)
    pad_masked_dst_ck = pad_sequence(masked_dst_ck, batch_first=True)
    pad_o_hour = pad_sequence(o_hour, batch_first=True)
    pad_d_hour = pad_sequence(d_hour, batch_first=True)
    pad_masked_d_hour = pad_sequence(masked_d_h, batch_first=True)
    pad_ori_t = pad_sequence(ori_t, batch_first=True)
    pad_dst_t = pad_sequence(dst_t, batch_first=True)
    pad_ori_l = pad_sequence(ori_l, batch_first=True)
    pad_dst_l = pad_sequence(dst_l, batch_first=True)
    pad_ori_tag = pad_sequence(ori_tag, batch_first=True)
    pad_dst_tag = pad_sequence(dst_tag, batch_first=True)

    ori_rg = torch.LongTensor(ori_rg)
    dst_rg = torch.LongTensor(dst_rg)
    uid = torch.LongTensor(uid)
    query_start_poi = torch.stack(query_start_poi, dim=0).long()
    query_start_hour = torch.stack(query_start_hour, dim=0).long()
    query_end_poi = torch.stack(query_end_poi, dim=0).long()
    query_end_hour = torch.stack(query_end_hour, dim=0).long()
    query_len = torch.stack(query_len, dim=0).long()
    user_profile = torch.stack(user_profile, dim=0)
    query_vec = torch.stack(query_vec, dim=0)

    lens_ori = torch.tensor([len(seq) for seq in ori_ck], dtype=torch.long)
    lens_dst = torch.tensor([len(seq) for seq in dst_ck], dtype=torch.long)
    max_len_ori = pad_ori_ck.size(1)
    max_len_dst = pad_dst_ck.size(1)

    ori_pad = torch.arange(max_len_ori).unsqueeze(0).expand(len(ori_ck), max_len_ori) < lens_ori.unsqueeze(1)
    dst_pad = torch.arange(max_len_dst).unsqueeze(0).expand(len(dst_ck), max_len_dst) < lens_dst.unsqueeze(1)

    return (
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
        list(home_prompt_text),
    )
