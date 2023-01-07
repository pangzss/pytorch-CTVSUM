# pytorch-CTVSUM

This repository contains the official Pytorch implementation of the paper
> [**Contrastive Losses Are Natural Criteria for Unsupervised Video Summarization**](https://arxiv.org/abs/2211.10056)
> 
> Zongshang Pang, Yuta Nakashima, Mayu Otani, Hajime Nagahara
> 
> In WACV2023

## Installation
```shell
git clone https://github.com/pangzss/pytorch-CTVSUM.git
cd pytorch-CTVSUM
conda env create -f environment.yml
conda activate ctvsum
```

## Dataset preparation
We use three datasets in our paper: [**TVSum**](https://github.com/yalesong/tvsum), [**SumMe**](https://gyglim.github.io/me/vsum/index.html), and a random subset of [**Youtube8M**](https://research.google.com/youtube8m/).

TVSum and SumMe are used for training and evaluation, and Youtube8M is only used for training.

To prepare the datasets, 

1. Download raw videos from TVSum and SumMe and put them in ./data/raw
2. Download the extracted features from [**GoogleDrive**](https://drive.google.com/drive/folders/1ruIbB8LoJ1sbF_q_yihLuolE4JpEgK8G?usp=sharing) (GoogLeNet features for TVSum and SumMe, kindly provided by the authors of [**DRDSN**](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce), and quantized Inception features for Youtube8M).
3. Put eccv_* files in ./data/interim, and unzip selected_features.zip in ./data/interim/youtube8M/

## Training and Evaluation
### Evaluation with only pretrained features
1. In ./configs/aln_unif_config.yml, modify the dataset and evaluation settings, e.g.
```yaml
data:
  name: tvsum # summe
  setting: Canonical # Augmented/Transfer
```
2. Set
```yaml
is_raw: True
```
3. Set use Global Consistency or not
```yaml
use_unif: True # False
```
4. For Youtube8M features (quantized Inception), in ./configs/aln_unif_y8_config.yml, set
```yaml
is_raw: True
hparams:
  use_unif: True # False
```
5. Run
```shell
./run_ablation.sh
./run_y8.sh
```
### Contrastive refinement and evaluation
1. For TVSum and SumMe, set training/evaluation setting in ./configs/aln_unif_config.yml, and decide whether to use global consistency or uniqueness filter
```yaml
is_raw: False
use_unif: True # False
use_unq: True # False
```
  The code will run 5-fold cross validation by default.
  
2. For Youtube8M, similarly in ./configs/aln_unif_y8_config.yml,
```yaml
is_raw: False
hparams:
  use_unif: True # False
  use_unq: True # False
```
3. Run
```bash
./run_ablation.sh
./run_y8.sh
```
## Acknowledgement
We would like to thank [**DRDSN**](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce), which provides the extracted features and the evaluation code for TVSum and SumMe. Moreover, we are thankful to the insightful work [**Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**](https://arxiv.org/abs/2005.10242), which inspired our work.

## Citation
```bibtex
@inproceedings{pang2023contrastive,
  title={Contrastive Losses Are Natural Criteria for Unsupervised Video Summarization},
  author={Pang, Zongshang and Nakashima, Yuta and Otani, Mayu and Nagahara, Hajime},
  booktitle={WACV},
  year={2023}
}
```
