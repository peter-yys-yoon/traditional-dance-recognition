
DATA_ROOT=/home/peter/workspace/dataset/kor_dance
VIDEO_DIR_PATH=$DATA_ROOT/videos
LIST_OUTPUT_PATH=/home/peter/dataset/kor_dance
    
DATASET=kordance600_13
python train.py --datadir $DATA_ROOT/$DATASET --threed_data --dataset $DATASET --frames_per_group 2 --groups 32  --backbone_net i3d_resnet -d 18 -b 8 -j 32 \
--lr 0.01 --logdir snapshots/ --epochs 50 --lr_steps 30 60 90

# DATASET=kordance600_39
# python train.py --datadir $DATA_ROOT/$DATASET --threed_data --dataset $DATASET --frames_per_group 2 --groups 32  --backbone_net i3d_resnet -d 18 -b 8 -j 32 \
# --lr 0.01 --logdir snapshots/ --epochs 100 --lr_steps 30 60 90

# DATASET=kordance800_13
# python train.py --datadir $DATA_ROOT/$DATASET --threed_data --dataset $DATASET --frames_per_group 2 --groups 32  --backbone_net i3d_resnet -d 18 -b 8 -j 32 \
# --lr 0.01 --logdir snapshots/ --epochs 100 --lr_steps 30 60 90

# DATASET=kordance800_78
# python train.py --datadir $DATA_ROOT/$DATASET --threed_data --dataset $DATASET --frames_per_group 2 --groups 32  --backbone_net i3d_resnet -d 18 -b 8 -j 32 \
# --lr 0.01 --logdir snapshots/ --epochs 100 --lr_steps 30 60 90