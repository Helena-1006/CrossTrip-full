import os
import pickle
from copy import copy
from collections import defaultdict, Counter
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    import pytz
except Exception:
    pytz = None


def convert_timestamp(region, timestamp, city_tz_mapping):
    ts_sec = int(float(timestamp))
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    if pytz is not None:
        local_tz = pytz.timezone(city_tz_mapping.get(region, "UTC"))
        local_dt = dt_utc.astimezone(local_tz)
    else:
        local_dt = dt_utc
    return local_dt, local_dt.hour, local_dt.weekday()


class TravelDatasetV2(Dataset):
    def __init__(self, args, ori_data_path, dst_data_path, trans_data_path):
        self.args = args
        self.dataset_dir = os.path.abspath(os.path.join(os.path.dirname(trans_data_path)))

        ori_raw = self._read_tsv_rows(ori_data_path)
        dst_raw = self._read_tsv_rows(dst_data_path)
        trans_raw = self._read_tsv_rows(trans_data_path)

        ori_events = [self._parse_visit_row(row) for row in ori_raw]
        dst_events = [self._parse_visit_row(row) for row in dst_raw]
        trans_events = [self._parse_trans_row(row) for row in trans_raw]
        ori_events = [x for x in ori_events if x is not None]
        dst_events = [x for x in dst_events if x is not None]
        trans_events = [x for x in trans_events if x is not None]

        self.poi_idx = {}
        self.tag_idx = {}
        self.region_idx = {}
        self.inv_region_idx = {}
        self.uids = []
        self.trans = []

        for row in trans_events:
            uid = row['uid']
            ori_region = row['ori_region']
            dst_region = row['dst_region']
            if ori_region not in self.region_idx:
                self.region_idx[ori_region] = len(self.region_idx)
                self.inv_region_idx[self.region_idx[ori_region]] = ori_region
            if dst_region not in self.region_idx:
                self.region_idx[dst_region] = len(self.region_idx)
                self.inv_region_idx[self.region_idx[dst_region]] = dst_region
            self.uids.append(uid)
            self.trans.append((self.region_idx[ori_region], self.region_idx[dst_region]))

        self.region_sample_count_tensor = torch.zeros(len(self.region_idx), dtype=torch.float)
        for _ori_rg, dst_rg in self.trans:
            self.region_sample_count_tensor[dst_rg] += 1.0

        bid_counter = Counter([r['bid'] for r in (ori_events + dst_events)])
        tag_counter = Counter([r['std_tag'] for r in (ori_events + dst_events)])
        self.poi_idx = {bid: idx for idx, (bid, _) in enumerate(bid_counter.most_common(), start=1)}
        self.tag_idx = {tag: idx for idx, (tag, _) in enumerate(tag_counter.most_common(), start=1)}
        self.inv_tag_idx = {v: k for k, v in self.tag_idx.items()}

        # Map model poi id -> category tree text for prompt enhancement.
        self.poi_id_to_cat_tree = {}
        for row in (ori_events + dst_events):
            bid = row['bid']
            cat_tree = row['cat_tree']
            poi_id = self.poi_idx.get(bid, 0)
            if poi_id > 0 and poi_id not in self.poi_id_to_cat_tree and len(cat_tree) > 0:
                self.poi_id_to_cat_tree[poi_id] = cat_tree

        city_tz_mapping = self._load_or_default_city_tz()
        poi_coord = self._load_or_default_poi_coord(ori_events, dst_events)
        self.poi_coord_norm = self._normalize_coord(poi_coord)

        self.poi_coord_tensor = torch.zeros((len(self.poi_idx) + 1, 2), dtype=torch.float)
        for bid, poi_id in self.poi_idx.items():
            coord = self.poi_coord_norm.get(bid, np.array([0.0, 0.0], dtype=np.float32))
            self.poi_coord_tensor[poi_id] = torch.from_numpy(coord)

        self.poi_popularity = torch.zeros(len(self.poi_idx) + 1, dtype=torch.float)
        for bid, cnt in bid_counter.items():
            self.poi_popularity[self.poi_idx[bid]] = cnt
        self.poi_popularity = torch.log1p(self.poi_popularity)

        home_by_uid = defaultdict(list)
        tour_by_uid = defaultdict(list)

        for row in ori_events:
            uid = row['uid']
            rid = row['rid']
            bid = row['bid']
            timestamp = row['timestamp']
            std_tag = row['std_tag']
            _, local_hour, local_weekday = convert_timestamp(rid, timestamp, city_tz_mapping)
            poi = self.poi_idx[bid]
            tag = self.tag_idx[std_tag]
            coord = self.poi_coord_norm.get(bid, np.array([0.0, 0.0], dtype=np.float32))
            home_by_uid[uid].append((poi, tag, float(timestamp), local_hour, local_weekday, coord))

        for row in dst_events:
            uid = row['uid']
            rid = row['rid']
            bid = row['bid']
            timestamp = row['timestamp']
            std_tag = row['std_tag']
            _, local_hour, local_weekday = convert_timestamp(rid, timestamp, city_tz_mapping)
            poi = self.poi_idx[bid]
            tag = self.tag_idx[std_tag]
            coord = self.poi_coord_norm.get(bid, np.array([0.0, 0.0], dtype=np.float32))
            tour_by_uid[uid].append((poi, tag, float(timestamp), local_hour, local_weekday, coord))

        self.oris = []
        self.dsts = []
        self.user_profile = []
        self.query_vec = []
        self.home_prompt_text = []

        for sample_uid, (ori_rg, dst_rg) in zip(self.uids, self.trans):
            home_traj = sorted(home_by_uid[sample_uid], key=lambda x: x[2])
            tour_traj = sorted(tour_by_uid[sample_uid], key=lambda x: x[2])

            if len(home_traj) == 0:
                home_traj = [(0, 0, 0.0, 0, 0, np.array([0.0, 0.0], dtype=np.float32))]
            if len(tour_traj) == 0:
                tour_traj = [(0, 0, 0.0, 0, 0, np.array([0.0, 0.0], dtype=np.float32))]

            self.oris.append(self._norm_traj(home_traj))
            self.dsts.append(self._norm_traj(tour_traj))
            self.user_profile.append(self._build_user_profile(home_traj))
            self.query_vec.append(self._build_query_vec(tour_traj, ori_rg, dst_rg))
            self.home_prompt_text.append(self._build_prompt_text(home_traj, ori_rg, dst_rg))

    @staticmethod
    def _read_tsv_rows(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.rstrip('\n').split('\t') for line in f if len(line.strip()) > 0]

    @staticmethod
    def _parse_trans_row(row):
        if len(row) < 4:
            return None
        try:
            return {
                'uid': int(row[0]),
                'ori_region': row[2],
                'dst_region': row[3],
            }
        except Exception:
            return None

    @staticmethod
    def _parse_visit_row(row):
        if len(row) < 6:
            return None
        try:
            uid = int(row[0])
            rid = row[2]
            bid = row[3]
            timestamp = float(row[4])
            std_tag = row[5]
            cat_tree = row[7] if len(row) > 7 and len(row[7].strip()) > 0 else std_tag
            return {
                'uid': uid,
                'rid': rid,
                'bid': bid,
                'timestamp': timestamp,
                'std_tag': std_tag,
                'cat_tree': cat_tree,
            }
        except Exception:
            return None

    def _build_prompt_text(self, home_traj, ori_rg, dst_rg):
        ori_city = self.inv_region_idx.get(int(ori_rg), f"region_{ori_rg}")
        dst_city = self.inv_region_idx.get(int(dst_rg), f"region_{dst_rg}")

        max_tokens = int(getattr(self.args, 'llm_max_traj_tokens', 64))
        traj = home_traj[:max_tokens]

        tags = [self.inv_tag_idx.get(int(x[1]), f"tag_{x[1]}") for x in traj]
        hours = np.array([int(x[3]) for x in traj], dtype=np.int32)
        weekends = np.array([int(x[4] >= 5) for x in traj], dtype=np.int32)

        def _bucket(h):
            if 6 <= h < 12:
                return 'morning'
            if 12 <= h < 18:
                return 'afternoon'
            if 18 <= h < 24:
                return 'evening'
            return 'night'

        time_buckets = [_bucket(int(h)) for h in hours.tolist()]
        bucket_counter = Counter(time_buckets)
        tag_counter = Counter(tags)
        top_tags = ', '.join([f"{k}({v})" for k, v in tag_counter.most_common(8)])
        top_time = ', '.join([f"{k}({v})" for k, v in bucket_counter.most_common()])

        prefix = (
            f"Task: infer transferable tourism preference from home-city trajectory. "
            f"Home city={ori_city}; Target city={dst_city}. "
            f"Prefer long-term leisure interests; downweight commute-only intent."
        )

        stats = (
            f"Summary: total_visits={len(traj)}; unique_tags={len(set(tags))}; "
            f"weekend_ratio={float(weekends.mean()) if len(weekends)>0 else 0.0:.2f}; "
            f"top_tags=[{top_tags}]; time_profile=[{top_time}]."
        )

        events = []
        cat_counter = Counter()
        for poi_id, tag_id, ts, hour, weekday, _coord in traj[:24]:
            tag_name = self.inv_tag_idx.get(int(tag_id), f"tag_{tag_id}")
            cat_tree = self.poi_id_to_cat_tree.get(int(poi_id), tag_name)
            cat_counter[cat_tree] += 1
            events.append(f"{tag_name} ({cat_tree})@{int(hour)}h")
        event_str = ' -> '.join(events) if len(events) > 0 else 'EMPTY'
        top_cat_trees = ', '.join([f"{k}({v})" for k, v in cat_counter.most_common(6)])

        suffix = "Output requirement: a compact semantic representation for recommendation model internal use."
        return f"{prefix}\n{stats}\nCategory trees: [{top_cat_trees}]\nRecent events: {event_str}\n{suffix}"

    def _load_or_default_city_tz(self):
        city_tz_file = os.path.join(self.dataset_dir, 'city_tz_mapping.pkl')
        if os.path.exists(city_tz_file):
            with open(city_tz_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def _load_or_default_poi_coord(self, ori_raw, dst_raw):
        poi_coord_file = os.path.join(self.dataset_dir, 'poi_coord.pkl')
        if os.path.exists(poi_coord_file):
            with open(poi_coord_file, 'rb') as f:
                return pickle.load(f)

        poi_coord = {}
        for row in (ori_raw + dst_raw):
            bid = row['bid']
            if bid not in poi_coord:
                poi_coord[bid] = (0.0, 0.0)
        return poi_coord

    @staticmethod
    def _normalize_coord(poi_coord):
        all_lats = [coord[0] for coord in poi_coord.values()]
        all_lons = [coord[1] for coord in poi_coord.values()]
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_range = lat_max - lat_min if lat_max != lat_min else 1.0
        lon_range = lon_max - lon_min if lon_max != lon_min else 1.0

        ret = {}
        for poi, (lat, lon) in poi_coord.items():
            ret[poi] = np.array([(lat - lat_min) / lat_range, (lon - lon_min) / lon_range], dtype=np.float32)
        return ret

    @staticmethod
    def _norm_traj(traj):
        timestamps = np.array([x[2] for x in traj], dtype=np.float32)
        t_min, t_max = timestamps.min(), timestamps.max()
        t_range = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
        norm_times = (timestamps - t_min) / t_range

        out = []
        for i, (poi, tag, ts, hour, weekday, coord) in enumerate(traj):
            out.append((poi, tag, ts, hour, weekday, float(norm_times[i]), coord))
        return out

    @staticmethod
    def _build_user_profile(home_traj):
        pois = [x[0] for x in home_traj]
        tags = [x[1] for x in home_traj]
        hours = np.array([x[3] for x in home_traj], dtype=np.float32)
        weekdays = np.array([x[4] for x in home_traj], dtype=np.float32)
        seq_len = len(home_traj)
        uniq_poi_ratio = len(set(pois)) / max(seq_len, 1)
        uniq_tag_ratio = len(set(tags)) / max(seq_len, 1)
        avg_hour = hours.mean() / 24.0
        std_hour = hours.std() / 24.0
        weekend_ratio = float(np.mean((weekdays >= 5).astype(np.float32)))
        repeat_ratio = 1.0 - uniq_poi_ratio
        len_norm = min(seq_len / 100.0, 1.0)
        return torch.tensor(
            [len_norm, uniq_poi_ratio, uniq_tag_ratio, avg_hour, std_hour, weekend_ratio, repeat_ratio],
            dtype=torch.float,
        )

    def _build_query_vec(self, tour_traj, ori_rg, dst_rg):
        hours = np.array([x[3] for x in tour_traj], dtype=np.float32)
        seq_len = len(tour_traj)
        target_len_norm = min(seq_len / 30.0, 1.0)
        poi_denom = float(max(self.poi_num - 1, 1))
        start_poi = (float(tour_traj[0][0]) / poi_denom) if len(tour_traj) > 0 else 0.0
        end_poi = (float(tour_traj[-1][0]) / poi_denom) if len(tour_traj) > 0 else 0.0
        start_hour = (hours[0] / 24.0) if len(hours) > 0 else 0.0
        end_hour = (hours[-1] / 24.0) if len(hours) > 0 else 0.0
        return torch.tensor(
            [start_poi, start_hour, end_poi, end_hour, target_len_norm, float(ori_rg), float(dst_rg)],
            dtype=torch.float,
        )

    @property
    def poi_num(self):
        return len(self.poi_idx) + 1

    @property
    def tag_num(self):
        return len(self.tag_idx) + 1

    @property
    def region_num(self):
        return len(self.region_idx)

    def __len__(self):
        return len(self.trans)

    def __getitem__(self, index):
        uid = self.uids[index]
        o = self.oris[index]
        d = self.dsts[index]
        ori_rg, dst_rg = self.trans[index]

        ori_ck = torch.LongTensor([x[0] for x in o])
        dst_ck = torch.LongTensor([x[0] for x in d])
        ori_tag = torch.LongTensor([x[1] for x in o])
        dst_tag = torch.LongTensor([x[1] for x in d])

        o_hour = torch.LongTensor([x[3] for x in o])
        d_hour = torch.LongTensor([x[3] for x in d])
        o_hour[o_hour == 0] = 24
        d_hour[d_hour == 0] = 24

        o_t = torch.FloatTensor([x[5] for x in o])
        d_t = torch.FloatTensor([x[5] for x in d])
        o_l = torch.from_numpy(np.stack([x[6] for x in o], axis=0)).float()
        d_l = torch.from_numpy(np.stack([x[6] for x in d], axis=0)).float()

        d_mask_indices = torch.arange(1, max(len(dst_ck) - 1, 1))
        masked_d_ck = dst_ck.clone()
        masked_d_h = d_hour.clone()
        if len(dst_ck) > 2:
            masked_d_ck[d_mask_indices] = 0
            masked_d_h[d_mask_indices] = 0

        query_start_poi = dst_ck[0] if len(dst_ck) > 0 else torch.tensor(0, dtype=torch.long)
        query_start_hour = d_hour[0] if len(d_hour) > 0 else torch.tensor(0, dtype=torch.long)
        query_end_poi = dst_ck[-1] if len(dst_ck) > 0 else torch.tensor(0, dtype=torch.long)
        query_end_hour = d_hour[-1] if len(d_hour) > 0 else torch.tensor(0, dtype=torch.long)
        query_len = torch.tensor(max(len(dst_ck), 2), dtype=torch.long)

        return (
            uid,
            ori_ck,
            dst_ck,
            masked_d_ck,
            o_hour,
            d_hour,
            masked_d_h,
            o_t,
            d_t,
            o_l,
            d_l,
            ori_rg,
            dst_rg,
            ori_tag,
            dst_tag,
            query_start_poi,
            query_start_hour,
            query_end_poi,
            query_end_hour,
            query_len,
            self.user_profile[index],
            self.query_vec[index],
            self.home_prompt_text[index],
        )


def _build_pair_robust_split_indices(trans_by_pair, ratios, rng, singleton_to_train=True):
    train_indice, valid_indice, test_indice = [], [], []

    for _t, us in trans_by_pair.items():
        us_shuf = copy(us)
        rng.shuffle(us_shuf)
        us_len = len(us_shuf)

        if us_len <= 0:
            continue
        if us_len == 1:
            if singleton_to_train:
                train_indice.extend(us_shuf)
            else:
                test_indice.extend(us_shuf)
            continue

        train_offset = int(us_len * ratios[0])
        valid_offset = int(us_len * (ratios[0] + ratios[1]))

        train_offset = max(train_offset, 1)
        train_offset = min(train_offset, us_len - 1)

        valid_offset = max(valid_offset, train_offset)
        valid_offset = min(valid_offset, us_len - 1)

        train_indice.extend(us_shuf[:train_offset])
        valid_indice.extend(us_shuf[train_offset:valid_offset])
        test_indice.extend(us_shuf[valid_offset:])

    return train_indice, valid_indice, test_indice


def random_split(dataset, split_path, ratios=(0.8, 0.1, 0.1), seed=2050, args=None):
    trans_by_pair = defaultdict(list)
    for u, t in enumerate(dataset.trans):
        trans_by_pair[t].append(u)

    split_strategy = 'legacy'
    split_singleton_to_train = True
    if args is not None:
        split_strategy = str(getattr(args, 'split_strategy', 'legacy'))
        split_singleton_to_train = bool(getattr(args, 'split_singleton_to_train', 1))

    if os.path.exists(split_path):
        with open(split_path, 'rb') as file:
            loaded = pickle.load(file)

        if isinstance(loaded, dict):
            loaded_strategy = str(loaded.get('split_strategy', 'legacy'))
            loaded_seed = int(loaded.get('seed', seed))

            # Keep full backward compatibility: legacy strategy can load old split files directly.
            if (loaded_strategy == split_strategy) and (loaded_seed == int(seed)):
                train_indice = loaded['train_indices']
                valid_indice = loaded['valid_indices']
                test_indice = loaded['test_indices']
                return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice)

            print(
                f"[INFO] Existing split file strategy/seed mismatch "
                f"(file: strategy={loaded_strategy}, seed={loaded_seed}; "
                f"current: strategy={split_strategy}, seed={int(seed)}). Rebuilding split."
            )
        else:
            # Old list/tuple split format is treated as legacy-compatible only.
            if split_strategy == 'legacy':
                train_indice, valid_indice, test_indice = loaded
                return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice)

            print("[INFO] Existing split file is legacy list format but current strategy is non-legacy. Rebuilding split.")

    rng = np.random.default_rng(seed)
    if split_strategy == 'pair_robust':
        train_indice, valid_indice, test_indice = _build_pair_robust_split_indices(
            trans_by_pair,
            ratios,
            rng,
            singleton_to_train=split_singleton_to_train,
        )
    else:
        train_indice, valid_indice, test_indice = [], [], []
        for _t, us in trans_by_pair.items():
            us_shuf = copy(us)
            rng.shuffle(us_shuf)
            us_len = len(us_shuf)
            train_offset = int(us_len * ratios[0])
            valid_offset = int(us_len * (ratios[0] + ratios[1]))
            train_indice.extend(us_shuf[:train_offset])
            valid_indice.extend(us_shuf[train_offset:valid_offset])
            test_indice.extend(us_shuf[valid_offset:])

    with open(split_path, 'wb') as file:
        payload = {
            'seed': int(seed),
            'ratios': tuple(float(x) for x in ratios),
            'split_strategy': split_strategy,
            'split_singleton_to_train': int(split_singleton_to_train),
            'dataset_size': int(len(dataset)),
            'train_indices': train_indice,
            'valid_indices': valid_indice,
            'test_indices': test_indice,
        }
        pickle.dump(payload, file)

    return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice)
