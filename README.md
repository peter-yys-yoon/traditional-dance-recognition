# Action Recognition Study

This repository is based on [IBM Action Recognition project](https://github.com/IBM/action-recognition-pytorch).


## Requirements
```
pip install -r requirement.txt
pip install scikit-video opencv-python
conda install ffmpeg=4.2.2
```

## Data Preparation

### Prepare raw videos.

Download vidoes and put them in the following directory format
```
-- DATA_ROOT
---- videos
------ video1.mp4
------ video2.mp4
------ ...
```

### Create a direcotry for the DATASET

make directory for DATASET and put splis in it.

```
mkdir kordance600_13
mv *.txt ./kordance600_13

-- DATA_ROOT
---- kordance600_13
------ train_list.txt
------ val_list.txt
------ class_ind.txt
```

### Extrating frames
Extract frames and make labels.

```
python tools/extract_video_kordance.py -i DATAROOT  -d DATASET
```

for example, if dataset is dataset is `kordance600_13`
```
python tools/extract_videos_kordance.py -i $DATA_ROOT -d kordance600_13
```

*Notethat scale reduction performed ffmpeg command at line 112*

The results will be as follows:

```
-- DATA_ROOT
---- kordance600_13
------ train_list.txt
------ val_list.txt
------ class_ind.txt
------ train
-------- video_0_folder
---------- 00001.jpg
---------- ...
-------- video_1_folder
------ val.txt
------ val
-------- video_0_folder
---------- 00001.jpg

```

Each line in `train.txt` and `val.txt` includes 4 elements and separated by a symbol, e.g. space or semicolon. 
Four elements (in order) include (1)relative paths to `video_x_folder` from `dataset_dir`, (2) starting frame number, usually 1, (3) ending frame number, (4) label id (a numeric number).

E.g., a `video_x` has `300` frames and belong to label `1`.
```
path/to/video_x_folder 1 300 1
```

## Training and Evaluation
Before training update custom dataset configuration in utils/dataset_config.py

The `opts.py` illustrates the available options for training 2D and 3D models. Some options are only for 2D models or 3D models.


Here is an example to train a `64-frame I3D` on the `Kinetics400` datasets with `Uniform Sampling` as input.

### Training
```
DATA_ROOT=/home/peter/dataset/kor_dance
DATASET=kordance600_13

python train.py --datadir $DATA_ROOT/$DATASET \
    --threed_data --dataset $DATASET --frames_per_group 2 \ --groups 32  --backbone_net i3d_resnet -d 18 -b 8 -j 32 \
    --lr 0.01 --logdir snapshots/ --epochs 150 --lr_steps 30 60 90
```
### Evaluation
```
DATA_ROOT=/home/peter/dataset/kor_dance
DATASET=$1
python test.py \
 --frames_per_group 2 --groups 32 --threed_data --backbone_net i3d_resnet -d 18 -b 4 -j 32 \
 -e --logdir logs/ --datadir $DATA_ROOT/$DATASET \
 --dataset $DATASET  --num_clips $3  --num_crops $4 \
 --pretrained $2



./unit_eval.sh kordance600_13 ./snapshots/kordance600_13-rgb-i3d-resnet-18-ts-f32-cosine-bs8-e150/model_best.pth.tar 3 3

```