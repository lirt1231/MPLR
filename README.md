# MPLR: a novel model for multi-target learning of logical rules for knowledge graph reasoning

This is the implementation of the model MPLR, proposed in the paper [MPLR: incorporating entity embedding into logic rule learning for knowledge graph reasoning](https://arxiv.org/abs/2112.06189).

Yuliang Wei, Haotian Li, Guodong Xin, Yao Wang, Bailing Wang

## Environment setup

- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy
- tqdm

It is recommended to create a conda virtual environment using the `requirements.yaml` by the following command:

```shell
conda create --name MPLR --file requirements.yaml
```

## Quick start

We provide a demo for training and evaluating our model on the `Family` dataset. All datasets can be found in the `datasets` folder.

### Training

Run the following command in you shell **under the root directory** of this repository to train a model.

```shell
python model/main.py --dataset=family
```

You may check the configuration file `model/configure.py` for more possible hyperparameter combinations. There is also a jupyter notebook for training this model in an interactive way locating at `model/train.ipynb`.

When the training process finishes, there are extra files created by the script that are stored under the directory `saved/family`, e.g., `option.txt` contains hyperparameters in this experiment and `prediction.txt` is the prediction results on test data for computing the metrics MRR (Mean Reciprocal Rank) and Hit@k.

### Evaluation

1. MRR & Hit@k

There is a separate script `eval/evaluate.py` to compute the MRR and Hit@k under the filtered protocol proposed in [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf), and you will see the evaluation result in your CLI.

```shell
python eval/evaluate.py --dataset=family --top_k=10 --rel
```

The last argument `rel` allows the script to compute MRR and Hit@k for each relation and print information on the CLI.

2. Mined rules and saturations

You can run the notebook `model/train.ipynb` to train a MPLR model as well as generate logic rules on the certain dataset.

The computation of saturations can be accessed in two notebooks, `datasets/graph_assessment.ipynb` and `datasets/graph_assessment-multihop.ipynb`, the former for saturations of rules with fixed length of two while the latter one allows varied lengths no longer than $L$.

For more details, please check the jupyter notebooks mentioned above.

## Citation

If you find this repository useful, please cite our [paper](https://arxiv.org/abs/2112.06189):

```
@article{wei2021mplr,
  title={MPLR: a novel model for multi-target learning of logical rules for knowledge graph reasoning},
  author={Wei, Yuliang and Li, Haotian and Xin, Guodong and Wang, Yao and Wang, Bailing},
  journal={arXiv preprint arXiv:2112.06189},
  year={2021}
}
```



