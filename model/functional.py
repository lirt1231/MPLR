#!/usr/bin/env python
# -*-coding:utf-8 -*-
# @file    :   functional.py
# @brief   :   Utility functions definition.
# @author  :   Haotian Li
# @email   :   lcyxlihaotian@126.com

from typing import List

import numpy as np
import torch


def get_recall(targets: torch.Tensor,
               predictions: torch.Tensor
               ) -> List[float]:
    """
    Args:
        `targets`: (batch_size, classes)
        `predictions`: (batch_size, classes)

    Returns:
        recall_list: (batch_size, ), recall score
    """
    targs = [
        set(np.argwhere(targ.numpy() > 0).reshape((-1, )))
        for targ in targets
    ]
    preds = [
        set(pred.argsort(descending=True)[:len(targs[bid])].tolist())
        for bid, pred in enumerate(predictions)
    ]
    recall_list = [
        len(targs[bid] & preds[bid]) / len(targs[bid])
        for bid in range(len(targs))
    ]

    return recall_list


def in_top_k(targets: torch.LongTensor,
             preds: torch.Tensor,
             k: int
             ) -> list:
    """Return is `targets` in topk of `preds`.

    Args:
        `targets`: (batch_size, )
        `preds`: (batch_size, classes)

    Returns:
        (batch_size, )
    """
    topk = preds.topk(k)[1]  # topk returns (values, indices)
    return (targets.unsqueeze(1) == topk).any(dim=1).tolist()


def in_top_k_multi(targets: torch.Tensor,
                   predictions: torch.Tensor,
                   k: int
                   ) -> List[bool]:
    """Return is `targets` in topk of `predictions`.

    Args:
        `targets`: (batch_size, )
        `predictions`: (batch_size, classes)

    Returns:
        (num_tails, )
    """
    res = []
    topk = predictions.topk(k)[1]  # topk() returns (values, indices)
    for bid in range(targets.size(0)):
        preds = set(topk[bid].numpy())
        targs = np.argwhere(targets[bid].numpy() > 0).reshape((-1, ))
        for targ in targs:
            if targ in preds:
                res.append(True)
            else:
                res.append(False)

    return res


def get_prediction(target: int,
                   predictions: np.ndarray,
                   id2ent: dict
                   ) -> List[List]:
    """Get entities that rank higher that the answer entity (tail)."""
    p_target = predictions[target]
    # Get entities whose logit is larger than that of target.
    pred = filter(lambda x: x[1] > p_target, enumerate(predictions))
    pred = sorted(pred, key=lambda x: x[1], reverse=True)
    pred.append((target, p_target))
    pred = [id2ent[targ] for targ, _ in pred]

    return pred


def get_prediction_multi(targets: np.ndarray,
                         predictions: np.ndarray,
                         id2ent: dict
                         ) -> List[List]:
    """Get entities that rank higher that the answer entity (tail) for multi-target reasoning."""
    res_preds = []
    target_indices = np.argwhere(targets > 0).reshape((-1, ))
    for target in target_indices:
        p_target = predictions[target]
        # Get entities whose logit is larger than that of target.
        pred = filter(lambda x: x[1] > p_target, enumerate(predictions))
        pred = sorted(pred, key=lambda x: x[1], reverse=True)
        pred.append((target, p_target))
        pred = [id2ent[targ] for targ, _ in pred]
        res_preds.append(pred)

    return res_preds
