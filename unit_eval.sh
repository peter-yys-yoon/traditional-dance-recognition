DATA_ROOT=/home/peter/workspace/dataset/kor_dance
WEIGHT=./snapshots/kordance600_13-rgb-i3d-resnet-18-ts-f32-cosine-bs8-e16/model_best.pth.tar

DATASET=kordance600_13

python video_demo.py \
 --frames_per_group 2 --groups 32 --threed_data --backbone_net i3d_resnet -d 18 -b 4 -j 32 \
 -e --logdir logs/ --datadir $DATA_ROOT/$DATASET \
 --dataset $DATASET  --num_clips 1  --num_crops 1 \
 --pretrained $WEIGHT --video A105C003B002D002P004S008
