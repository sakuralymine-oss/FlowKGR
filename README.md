# FMGNN

Code for transductive and fully inductive knowledge graph reasoning with a query-conditioned GNN, candidate denoising VAE, and score flow matching.

## Dependencies

- python == 3.10
- torch == 1.12.1
- torch_scatter == 2.0.9
- numpy == 1.21.6
- scipy == 1.10.1

CUDA is required. The PyTorch and torch-scatter builds must match the installed CUDA version.

## Reproduction

Dataset-specific hyperparameters are configured in `train.py`. Additional arguments are only needed to override the defaults.

### Transductive settings (in `transductive`)

#### Family dataset

```bash
python train.py ./data/family
```

#### UMLS dataset

```bash
python train.py ./data/umls
```

#### WN18RR dataset

```bash
python train.py ./data/WN18RR
```

#### FB15k-237 dataset

```bash
python train.py ./data/fb15k-237
```

#### NELL995 dataset

```bash
python train.py ./data/nell
```

#### YAGO3-10 dataset

```bash
python train.py ./data/YAGO
```

### Inductive settings (in `inductive`)

#### WN18RR datasets

```bash
python train.py ./data/WN18RR_v1 --inductive
python train.py ./data/WN18RR_v2 --inductive
python train.py ./data/WN18RR_v3 --inductive
python train.py ./data/WN18RR_v4 --inductive
```

#### FB237 datasets

```bash
python train.py ./data/fb237_v1 --inductive
python train.py ./data/fb237_v2 --inductive
python train.py ./data/fb237_v3 --inductive
python train.py ./data/fb237_v4 --inductive
```

#### NELL995 datasets

```bash
python train.py ./data/nell_v1 --inductive
python train.py ./data/nell_v2 --inductive
python train.py ./data/nell_v3 --inductive
python train.py ./data/nell_v4 --inductive
```

## Evaluation

```bash
python train.py ./data/WN18RR --checkpoint ./data/WN18RR/saveModel/model.pt --eval_only --eval_split both
```

Add `--inductive` when evaluating a fully inductive checkpoint.

## Output

Training prints one line per epoch:

```text
[EPOCH 1] LOSS:1.2345  [VALID] MRR:0.1234 MR:50.2 H@1:0.0800 H@3:0.1400 H@10:0.2500
```

The best validation checkpoint is saved under the dataset `saveModel` directory. Final results are written to the configured result file.
