import os
import random
import numpy as np
import pandas as pd

import torch
from config import CFG


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=CFG.device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt):
    """
    tgt: shape(N, L)
    """
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_padding_mask = (tgt == CFG.pad_idx)

    return tgt_mask, tgt_padding_mask


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def reformat_preds(preds_df, tokenizer):
    """
    preds_df: dataframe containing two columns -> ['id', 'preds']
    tokenizer: tokenizer object used in training
    """

    preds_df['pred'] = preds_df['pred'].map(
        lambda x: eval(x) if not pd.isna(x) else [])
    preds_df = preds_df.explode("pred")
    preds_df['pred'] = preds_df['pred'].map(
        lambda x: x if isinstance(x, list) else [-1]*6)

    preds = pd.DataFrame(preds_df['pred'].tolist(), columns=[
                         'xmin', 'ymin', 'xmax', 'ymax', 'label', 'conf'])
    preds[['xmin', 'ymin', 'xmax', 'ymax']] = preds[[
        'xmin', 'ymin', 'xmax', 'ymax']] / float(CFG.img_size)

    preds_df.reset_index(drop=True, inplace=True)
    preds.reset_index(drop=True, inplace=True)

    preds_df = pd.concat([preds_df, preds], axis=1)
    preds_df['label'] = preds_df['label'] - tokenizer.num_bins

    return preds_df
