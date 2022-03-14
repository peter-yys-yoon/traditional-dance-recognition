
##  kor dance 600 30 -------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------->>
# weight_base=./snapshots/kordance600_13-rgb-i3d-resnet-18-ts-f32-cosine-bs8-e150/checkpoint_
# for i in {1..15}
# do
#     rem=$(($i % 2))
#     if [ "$rem" -ne "0" ]; then
#         kk=$(printf "%03g" $i)
#         ./unit_eval.sh kordance600_13 $weight_base$kk.pth.tar 3 1
#     fi
# done

##  <<<<< ------------------------------------------------------------------------------------------



##  kor dance 600 39 -------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------->>
weight_base=./snapshots/kordance600_13-rgb-i3d-resnet-18-ts-f32-cosine-bs8-e150/checkpoint_
k=011.pth.tar
./unit_eval.sh kordance600_13 $weight_base$k 3 1
# for i in {1..50}
# do
#     rem=$(($i % 2))
#     if [ "$rem" -ne "0" ]; then
#         kk=$(printf "%03g" $i)
#         ./unit_eval.sh kordance600_39 $weight_base$kk.pth.tar 3 1
#     fi
# done

##  <<<<< ------------------------------------------------------------------------------------------








# weight_base=./snapshots/kordance600_39-rgb-i3d-resnet-18-ts-f32-cosine-bs8-e100/checkpoint_
# for i in $(seq -f "%03g" 30 81)
# do
#     ./unit_eval.sh kordance600_13 $weight_base$i.pth.tar 3 1
# done


# weight_base=./snapshots/kordance800_78-rgb-i3d-resnet-18-ts-f32-cosine-bs8-e100/checkpoint_
# for i in $(seq -f "%03g" 30 81)
# do
#     ./unit_eval.sh kordance600_13 $weight_base$i.pth.tar 3 1
# done











# ./unit_eval.sh kordance600_13 $weight 2 3
# ./unit_eval.sh kordance600_13 $weight 1 3