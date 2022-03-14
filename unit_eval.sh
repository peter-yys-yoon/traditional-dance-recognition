DATA_ROOT=/home/peter/dataset/kor_dance
DATASET=$1
python test.py \
 --frames_per_group 2 --groups 32 --threed_data --backbone_net i3d_resnet -d 18 -b 4 -j 32 \
 -e --logdir logs/ --datadir $DATA_ROOT/$DATASET \
 --dataset $DATASET  --num_clips $3  --num_crops $4 \
 --pretrained $2
