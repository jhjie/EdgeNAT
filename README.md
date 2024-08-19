# EdgeNAT: Transformer for Efficient Edge Detection
In this paper, we propose EdgeNAT, a one-stage Transformer-based edge detector that effectively extracts object boundaries and meaningful edges. Our model leverages image global contextual information and detailed local cues by exploiting the multi-scale structure of the backbone network. Here are the code for this paper.

## Requirements
```
pip3 install -r requirements-base.txt  # Installs torch 
pip3 install -r requirements.txt # Installs NATTEN, MMCV, MMSEG, and others
```
## Initial weights
You can download the initial weights from [hear](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/segmentation/DiNAT.md). The five .pth files of pre-training should be placed in the folder ./pretrained.

[DiNAT-Mini](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth)

[DiNAT-Tiny](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth)

[DiNAT-Small](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth)

[DiNAT-Base](https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth)

[DiNAT-Large](https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth)
## Training
### Training on BSDS500:
```
./tools/dist_train.sh configs/bsds/edgenat_l_320x320_40k_bsds.py 1   # EdgeNAT-L
./tools/dist_train.sh configs/bsds/edgenat_s0_320x320_40k_bsds.py 1  # EdgeNAT-S0
./tools/dist_train.sh configs/bsds/edgenat_s1_320x320_40k_bsds.py 1  # EdgeNAT-S1
./tools/dist_train.sh configs/bsds/edgenat_s2_320x320_40k_bsds.py 1  # EdgeNAT-S2
./tools/dist_train.sh configs/bsds/edgenat_s3_320x320_40k_bsds.py 1  # EdgeNAT-S3
```
### Training on BSDS-VOC (BSDS500 and PASCAL VOC Context):
#### Step 1: Training on PASCAL VOC Context
```
./tools/dist_train.sh configs/bsds/edgenat_l_320x320_40k_bsds_pascal.py 1   # EdgeNAT-L
./tools/dist_train.sh configs/bsds/edgenat_s0_320x320_40k_bsds_pascal.py 1  # EdgeNAT-S0
./tools/dist_train.sh configs/bsds/edgenat_s1_320x320_40k_bsds_pascal.py 1  # EdgeNAT-S1
./tools/dist_train.sh configs/bsds/edgenat_s2_320x320_40k_bsds_pascal.py 1  # EdgeNAT-S2
./tools/dist_train.sh configs/bsds/edgenat_s3_320x320_40k_bsds_pascal.py 1  # EdgeNAT-S3
```
#### Step 2: Training on BSDS500
```
./tools/dist_train.sh configs/bsds/edgenat_l_320x320_40k_bsds.py 1 --load-from ./work_dirs/edgenat_l_320x320_40k_bsds_pascal/iter_XX000.pth  # EdgeNAT-L

./tools/dist_train.sh configs/bsds/edgenat_s0_320x320_40k_bsds.py 1 --load-from ./work_dirs/edgenat_s0_320x320_40k_bsds_pascal/iter_XX000.pth # EdgeNAT-S0

./tools/dist_train.sh configs/bsds/edgenat_s1_320x320_40k_bsds.py 1 --load-from ./work_dirs/edgenat_s1_320x320_40k_bsds_pascal/iter_XX000.pth # EdgeNAT-S1

./tools/dist_train.sh configs/bsds/edgenat_s2_320x320_40k_bsds.py 1 --load-from ./work_dirs/edgenat_s2_320x320_40k_bsds_pascal/iter_XX000.pth # EdgeNAT-S2

./tools/dist_train.sh configs/bsds/edgenat_s3_320x320_40k_bsds.py 1 --load-from ./work_dirs/edgenat_s3_320x320_40k_bsds_pascal/iter_XX000.pth # EdgeNAT-S3
```
## Evaluation
The code is evaluated on MATLAB R2018b.
### BSDS500:
```
cd eval
(echo "data_dir = '../output/epoch-x-test'"; cat eval_bsds.m)|matlab -nodisplay -nodesktop -nosplash
```
### NYUDv2:
```
cd eval
(echo "data_dir = '../output/epoch-x-test'"; cat eval_nyud.m)|matlab -nodisplay -nodesktop -nosplash
```
