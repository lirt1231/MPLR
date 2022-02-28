import argparse
import os
from collections import defaultdict

import numpy as np


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="family", type=str)
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--rel', default=False, action="store_true")
    option = parser.parse_args()
    print(option)

    preds = os.path.join("saved", option.dataset, "prediction.txt")

    # Read prediction file.
    lines = [line.strip().split(",") for line in open(preds).readlines()]
    line_cnt = len(lines)

    hits = 0
    hits_by_q = defaultdict(list)
    ranks = 0
    ranks_by_q = defaultdict(list)
    rranks = 0.
    line_cnt = 0

    for line in lines:
        assert(len(line) > 3)
        q, h, t = line[0:3]
        this_preds = set(line[3:])
        if h in this_preds and h != t:
            this_preds.remove(h)

        line_cnt += 1
        hitted = 0.
        if len(this_preds) <= option.top_k:
            hitted = 1.
        rank = len(this_preds)

        hits += hitted
        ranks += rank
        rranks += 1. / rank
        hits_by_q[q].append(hitted)
        ranks_by_q[q].append(rank)

    print("Hits at %d is %0.4f" % (option.top_k, hits / line_cnt))
    print("Mean rank %0.2f" % (1. * ranks / line_cnt))
    print("Mean Reciprocal Rank %0.4f" % (1. * rranks / line_cnt))

    # [k, np.mean(v), len(v)]: [relation, hit@k, num_triplets]
    if option.rel:
        hits_by_q_mean = sorted([[k, np.mean(v), len(v)]
                                for k, v in hits_by_q.items()], key=lambda xs: xs[1], reverse=True)
        for xs in hits_by_q_mean:
            xs += [np.mean(ranks_by_q[xs[0]]), np.mean(1. / np.array(ranks_by_q[xs[0]]))]
            # print: relation, hit@k, num_triplets, mean rank, MRR
            print(", ".join([str(x) for x in xs]))


if __name__ == "__main__":
    evaluate()
