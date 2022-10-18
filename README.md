# DiffAug: Differential Data Augmentation for Contrastive Sentence Representation Learning

Our code is based on the [SimCSE](https://github.com/princeton-nlp/SimCSE)

## Install dependencies

First, install PyTorch on [the official website](https://pytorch.org). All our experiments are conducted with PyTorch v1.8.1 with CUDA v10.1. So you may use the following code to download the same PyTorch version:

```bash
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Then run the following script to install the remaining dependencies,
```bash
pip install -r requirements.txt
```

## Prepare training and evaluation datasets

We use the same training and evaluation datasets as [SimCSE](https://github.com/princeton-nlp/SimCSE). Therefore, we adopt their scripts for downloading the datasets.

To download the unlabeled Wikipedia dataset, please run
```bash
cd data/
bash download_wiki.sh
```

To download the labeled NLI dataset, please run
```bash
cd data/
bash download_nli.sh
```

To download the evaluation datasets, please run
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```
Following previous works, we use [SentEval](https://github.com/facebookresearch/SentEval) to evaluate our model.

## Training

We prepared two example scripts for reproducing our results under the semi-supervised and supervised settings respectively.

To train a semi-supervised model, please run
```bash
bash run_semi_sup_bert.sh
```

To train a supervised model, please run
```bash
bash run_sup_bert.sh
```

After training, the results on the test set are expected to be printed out in a tabular format:
```
*** E.g. Supervised model evaluatation ***
+-------+-------+-------+-------+-------+--------+---------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 |  STSB  |  SICKR  |  Avg. |
+-------+-------+-------+-------+-------+--------+---------+-------+
| 77.40 | 85.24 | 80.50 | 86.85 | 82.59 | 84.12  |  80.29  | 82.43 |
+-------+-------+-------+-------+-------+--------+---------+-------+
```


