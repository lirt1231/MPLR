#!/usr/bin/env python
# -*-coding:utf-8 -*-
# @file    :   dataloader.py
# @author  :   Haotian Li
# @email   :   lcyxlihaotian@126.com

import json
import os
from tqdm import tqdm

import numpy as np


class RTDataLoader(object):
    def __init__(self,
                 relations_file: str,
                 entities_file: str,
                 all_file: str,
                 train_file: str,
                 valid_file: str,
                 test_file: str
                 ) -> None:
        # Read relations and entities.
        self.rel2id = self._read_relations_file(relations_file)
        self.id2rel = {ident: rel for rel, ident in self.rel2id.items()}
        self.num_relations = len(self.rel2id)
        self.num_operators = self.num_relations
        self.ent2id = self._read_entities_file(entities_file)
        self.id2ent = {ident: ent for ent, ident in self.ent2id.items()}
        self.num_entities = len(self.ent2id)

        # Read graph.
        self.train = self._parse_data_json(train_file)
        self.valid = self._parse_triplets(valid_file)
        self.test = self._parse_triplets(test_file)

        # Construct adjacency matrices.
        triplets_all = self._parse_triplets(all_file)
        triplets_train = self._convert_to_triplets(self.train)

        self.matrices_train = self._get_adjacency_matrices(triplets_all)
        self.matrices_valid = self._get_adjacency_matrices(
            np.concatenate([triplets_train, self.test])
        )
        self.matrices_test = self._get_adjacency_matrices(
            np.concatenate([triplets_train, self.valid])
        )

    def one_epoch(self,
                  name: str,
                  batch_size: int,
                  num_sample_batches: int = 0,
                  shuffle: bool = False
                  ) -> tuple:
        """Load batch data for one epoch train | valid | test.

        :Args
            `name`: 'train' | 'valid' | 'test'
            `batch_size`: mini batch size
            `num_sample_batches`: max number of batches for one epoch
            `shuffle`: shuffle data inside batch

        :Returns
            `batch_size` of queries and corresponding matrices.
        """
        if name not in ["train", "valid", "test"]:
            raise Exception("{} cannot be loaded.".format(name))
        if (name == "valid" and self.valid is None) or\
                (name == "test" and self.test is None):
            raise Exception("{} not loaded.".format(name))

        samples = getattr(self, name)
        num_samples = len(samples)
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        batch_cnt = 0
        for batch_start in range(0, num_samples, batch_size):
            batch_cnt += 1
            if num_sample_batches != 0 and\
                    batch_cnt >= num_sample_batches and\
                    name == "train":
                break
            ids = indices[batch_start: batch_start+batch_size]
            this_batch = samples[ids]
            # If evaluation, evaluated triplets should be removed from graph.
            if name == "train":
                queries, heads, tails = self._train_to_feed(this_batch)
                matrices = self.matrices_train
            else:
                queries, heads, tails = self._eval_to_feed(this_batch)
                matrices = getattr(self, "matrices_" + name)
                matrices = self._augment_matrices(
                    samples, this_batch, matrices
                )

            yield queries, heads, tails, matrices

    def _read_relations_file(self, relations_file: str):
        """Load relations and return relation-to-index map."""
        rel2id = {}
        with open(relations_file, 'r') as file:
            for line in tqdm(file, "Loading relations"):
                line = line.strip()
                rel2id[line] = len(rel2id)
        return rel2id

    def _read_entities_file(self, entities_file: str):
        """Load entities and return entity-to-index map."""
        ent2id = {}
        with open(entities_file, 'r') as file:
            for line in tqdm(file, "Loading entities"):
                line = line.strip()
                ent2id[line] = len(ent2id)
        return ent2id

    def _parse_triplets(self, triplets_file: str):
        """Read triplets (query, head, tail)."""
        triplets = []
        with open(triplets_file, 'r') as file:
            for line in tqdm(file, "Loading triplets"):
                line = line.strip().split('\t')
                assert(len(line) == 3)
                triplets.append(
                    (
                        self.rel2id[line[1]],
                        self.ent2id[line[0]],
                        self.ent2id[line[2]]
                    )
                )
        return np.array(triplets)

    def _parse_data_json(self, data_path: str):
        """Read data from json file."""
        parsed_data = []
        with open(data_path, 'r') as file:
            raw_data = json.load(file)
        for record in raw_data:
            head = self.ent2id[record['head']]
            for rel, tails in record['relations'].items():
                parsed_data.append(
                    {
                        'head': head,
                        'rel': self.rel2id[rel],
                        'tails': [self.ent2id[tail] for tail in tails]
                    }
                )
        return np.array(parsed_data, dtype=object)

    def _convert_to_triplets(self, data_json: np.ndarray):
        """Convert data loaded from json to triplets format."""
        triplets = [
            (data['rel'], data['head'], tail)
            for data in data_json for tail in data['tails']
        ]
        return np.array(triplets, dtype=object)

    def _get_adjacency_matrices(self, triplets: np.ndarray):
        """Get adjacency matrix from triplets in preparation for
           creating sparse matrix in PyTorch.
        """
        matrices = {
            r: ([[0, 0]], [0.], [self.num_entities, self.num_entities])
            for r in range(self.num_relations)
        }
        for rel, head, tail in triplets:
            value = 1.
            matrices[rel][0].append([head, tail])
            matrices[rel][1].append(value)
        return matrices

    def _combine_matrices(self, mat1: dict, mat2: dict):
        """Combine two sets of adjacency matrices.
           Assume mat1 and mat2 contain distinct elements.
        """
        new_matrix = {}

        for key, value in mat1.items():
            new_matrix[key] = value
        for key, value2 in mat2.items():
            try:
                value1 = mat1[key]
                new_matrix[key] = (
                    value1[0] + value2[0],
                    value1[1] + value2[1],
                    value1[2]
                )
            except KeyError:
                new_matrix[key] = value
        return new_matrix

    def _train_to_feed(self, samples: np.ndarray) -> tuple:
        """Separate training samples to queries, heads and tails."""
        queries, heads, tails = [], [], []
        for sample in samples:
            queries.append(sample['rel'])
            heads.append(sample['head'])
            tails.append(sample['tails'])
        return np.array(queries, dtype=np.int64),\
            np.array(heads, dtype=np.int64),\
            np.array(tails, dtype=object)

    def _eval_to_feed(self, triplets: np.ndarray) -> tuple:
        """Separate valid & test samples to queries, heads and tails."""
        queries, heads, tails = zip(*triplets)
        return np.array(queries, dtype=np.int64),\
            np.array(heads, dtype=np.int64),\
            np.array(tails, dtype=np.int64)

    def _augment_matrices(self,
                          all_triplets: np.ndarray,
                          sample_triplets: np.ndarray,
                          ori_matrices: dict
                          ) -> dict:
        """Augment existing matrices using extra triplets."""
        trips_this_batch = set()
        for q, h, t in sample_triplets:
            trips_this_batch.add((q, h, t))
        extra_triplets = [
            (q, h, t)
            for q, h, t in all_triplets
            if (q, h, t) not in trips_this_batch
        ]
        extra_matrices = self._get_adjacency_matrices(extra_triplets)
        aug_matrices = self._combine_matrices(
            extra_matrices, ori_matrices
        )

        return aug_matrices


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "../datasets/")
    dataset = "family"
    dataset_dir = os.path.join(data_dir, dataset)
    entities_file = os.path.join(dataset_dir, "entities.txt")
    relations_file = os.path.join(dataset_dir, "relations.txt")
    all_file = os.path.join(dataset_dir, "all.txt")
    train_file = os.path.join(dataset_dir, "train.json")
    valid_file = os.path.join(dataset_dir, "valid.txt")
    test_file = os.path.join(dataset_dir, "test.txt")

    dataloader = RTDataLoader(relations_file, entities_file,
                              all_file, train_file,
                              valid_file, test_file)

    for bid, (qq, hh, TT, mat) in enumerate(dataloader.one_epoch(
        "train", 2, False
    )):
        print(qq.shape)
        print(hh.shape)
        print(TT.shape)
        print(TT)
        print(list(mat.keys()))
        break
