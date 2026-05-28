import numpy as np
import torch


def count_adjacent_repetition_rate(input_data):
    if isinstance(input_data, list):
        predictions = input_data
    elif hasattr(input_data, 'numpy'):
        predictions = input_data.cpu().numpy().flatten().tolist()
    else:
        raise ValueError("Input data must be a list or a tensor.")

    total = len(predictions)
    if total < 2:
        return 0.0

    repeated = sum(1 for i in range(1, total) if predictions[i] == predictions[i - 1])
    repetition_ratio = repeated / (total - 1)
    return repetition_ratio


def f1_score(target, predict, noloop=False):
    assert isinstance(noloop, bool)
    assert len(target) > 0
    assert len(predict) > 0

    if noloop:
        intersize = len(set(target) & set(predict))
    else:
        match_tags = np.zeros(len(target), dtype=np.bool_)
        for poi in predict:
            for j in range(len(target)):
                if not match_tags[j] and poi == target[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(target)
    precision = intersize * 1.0 / len(predict)
    denom = recall + precision if (recall + precision) != 0 else 1.0
    return 2 * precision * recall / denom


def pairs_f1_score(target, predict):
    assert target.numel() > 0
    n = target.numel()
    nr = predict.numel()
    if n == 1 and nr == 1:
        return 1.0 if target.item() == predict.item() else 0.0
    if n == 1 or nr == 1:
        return 0.0

    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    order_dict = {}
    for i, poi in enumerate(target):
        order_dict[poi.item()] = i

    nc = 0
    for i in range(nr):
        poi1 = predict[i].item()
        for j in range(i + 1, nr):
            poi2 = predict[j].item()
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    if nc == 0:
        return 0.0
    precision = nc / n0r
    recall = nc / n0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_sequences(target_batch, pred_batch):
    batch_f1, batch_pairs_f1, batch_full_f1, batch_full_pairs_f1, batch_rep, batch_full_rep = [], [], [], [], [], []

    batch_size = target_batch.size(0)
    for i in range(batch_size):
        t_seq = [x for x in target_batch[i].cpu().numpy().tolist() if x != 0]
        p_seq = [x for x in pred_batch[i].cpu().numpy().tolist() if x != 0]
        if len(t_seq) < 2:
            continue

        target_len = len(t_seq)
        if len(p_seq) > target_len:
            p_seq = p_seq[:target_len]

        t_mid = t_seq[1:-1]
        p_mid = p_seq[1:-1] if len(p_seq) >= 2 else []

        t_full = t_seq
        p_full = [t_seq[0]] + p_mid + [t_seq[-1]]

        if len(t_mid) > 0 and len(p_mid) > 0:
            f1 = f1_score(t_mid, p_mid)
            p_f1 = pairs_f1_score(torch.LongTensor(t_mid), torch.LongTensor(p_mid))
        else:
            f1, p_f1 = 0.0, 0.0

        rep = count_adjacent_repetition_rate(p_mid)
        full_f1 = f1_score(t_full, p_full)
        full_p_f1 = pairs_f1_score(torch.LongTensor(t_full), torch.LongTensor(p_full))
        full_rep = count_adjacent_repetition_rate(p_full)

        batch_f1.append(f1)
        batch_pairs_f1.append(p_f1)
        batch_full_f1.append(full_f1)
        batch_full_pairs_f1.append(full_p_f1)
        batch_rep.append(rep)
        batch_full_rep.append(full_rep)

    return {
        "f1": float(np.mean(batch_f1)) if batch_f1 else 0.0,
        "pairs_f1": float(np.mean(batch_pairs_f1)) if batch_pairs_f1 else 0.0,
        "full_f1": float(np.mean(batch_full_f1)) if batch_full_f1 else 0.0,
        "full_pairs_f1": float(np.mean(batch_full_pairs_f1)) if batch_full_pairs_f1 else 0.0,
        "rep": float(np.mean(batch_rep)) if batch_rep else 0.0,
        "full_rep": float(np.mean(batch_full_rep)) if batch_full_rep else 0.0,
    }
