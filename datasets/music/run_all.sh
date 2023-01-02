#!/bin/bash
echo "music"

pwd

# uncomment to install if needed
# peerannot install ./../music/music.py
#
# for strat in MV NaiveSoft DS GLAD
# do
# echo "Strategy: ${strat}"
# peerannot aggregate ./../music/ -s $strat
# declare -l strat
# strat=$strat
# for i in `seq 1 5`
# do
# echo "Repetition ${i}"
# peerannot train ./../music -o music_${strat}_$i -K 10 --labels=./../music/labels/labels_music_${strat}.npy --model resnet18 --n-epochs=500 --lr=0.001 --scheduler -m 150 -m 400 --num-workers=8 --pretrained --img-size=224 --batch-size=64 --optimizer=adam
# done
# done
#
# echo "Getting AUM"
# peerannot identify ./../music -K 10 --method AUM \
#                     --model resnet18 \
#                     --n-epochs 100 --lr=0.001 --optimizer=adam \
#                     --pretrained


# echo "Strategy: WAUMstacked"
# peerannot identify ./../music/ -K 10 --method WAUMstacked --labels ./../music/answers.json --model resnet18 --n-epochs 500 --lr=0.001 --maxiter-DS=100 --alpha=0.05 --pretrained --optimizer=adam
# for i in `seq 1 5`
# do
# echo "Repetition ${i}"
# peerannot train ./../music -o music_waum_0.01 -K 10 --labels=./../music/labels/labels_waumstacked_0.01.npy --model resnet18 --n-epochs=500 --lr=0.001 --scheduler -m 150 -m 400 --num-workers=8 --path-remove ./../music/identification/waum_stacked_0.01_yang/too_hard_0.01.txt --pretrained --optimizer=adam --batch-size=64
# done
# #
# for strat in MV NaiveSoft DS GLAD
# do
# echo "Strategy: WAUM stacked + ${strat}"
# declare -l strat
# strat=$strat
# for i in `seq 1 5`
# do
# echo "Repetition ${i}"
# peerannot train ./../music -o music_waum_0.01_${strat}_${i} -K 10 --labels=./../music/labels/labels_waumstacked_0.01.npy --model resnet18 --n-epochs=500 --lr=0.001 --scheduler -m 150 -m 400 --num-workers=8 --path-remove ./../music/identification/waum_stacked_0.01_yang/too_hard_0.01.txt --pretrained --optimizer=adam --batch-size=64
# done
# done
#
# # for strat in CoNAL[scale=0] CoNAL[scale=1e-4] CrowdLayer[scale=0] CrowdLayer[scale=1e-4]
for strat in CrowdLayer[scale=0] CrowdLayer[scale=1e-4]
do
echo "Strategy: ${strat}"
declare -l strat
strat=$strat
for i in `seq 1 5`
do
peerannot aggregate-deep ./../music -o music_${strat}_${i} --answers ./../music/answers.json --model resnet18 --n-classes=10 --n-epochs 500 --lr 0.001 --optimizer adam --batch-size 64 --num-workers 8 --img-size=224 -s ${strat} --scheduler -m 150 -m 400 --pretrained
done
done
