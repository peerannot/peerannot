#!/bin/bash
echo "music"

pwd

# uncomment to install if needed
# peerannot install ./../music/music.py

for strat in MV NaiveSoft DS GLAD
do
echo "Strategy: ${strat}"
peerannot aggregate ./../music/ -s $strat
declare -l strat
strat=$strat
for i in `seq 1 1`
do
echo "Repetition ${i}"
peerannot train ./../music -o music_${strat}_$i -K 10 --labels=./../music/labels/labels_music_${strat}.npy --model modelmusic --n-epochs=2000 --lr=0.001 --scheduler -m 10000 --num-workers=8 --pretrained --img-size=224 --batch-size=64 --optimizer=adam --data-augmentation
done
done


echo "Strategy: WAUM"
peerannot identify ./../music/ -K 10 --method WAUM --labels ./../music/answers.json --model resnet50 --n-epochs 1000 --lr=0.001 --maxiter-DS=100 --alpha=0.05 --pretrained --optimizer=adam
for i in `seq 1 1`
do
echo "Repetition ${i}"
peerannot train ./../music -o music_waum_0.05_${i} -K 10 --labels=./../music/labels/labels_waum_0.05.npy --model modelmusic --n-epochs=2000 --lr=0.001 --scheduler -m 10000 --num-workers=8 --path-remove ./../music/identification/resnet50/waum_0.05_yang/too_hard_0.05.txt --pretrained --optimizer=adam --batch-size=64 --data-augmentation
done
# #
for strat in MV NaiveSoft DS GLAD
do
echo "Strategy: WAUM + ${strat}"
declare -l strat
strat=$strat
for i in `seq 1 1`
do
echo "Repetition ${i}"
peerannot train ./../music -o music_waum_0.05_${strat}_${i} -K 10 --labels=./../music/labels/labels_waum_0.05.npy --model modelmusic --n-epochs=2000 --lr=0.001 --scheduler -m 10000 --num-workers=8 --path-remove ./../music/identification/resnet50/waum_0.05_yang/too_hard_0.05.txt --pretrained --optimizer=adam --batch-size=64 --data-augmentation
done
done

echo "WAUM + CoNAL"
for strat in CoNAL[scale=0] CoNAL[scale=1e-4]
do
echo "Strategy: ${strat}"
declare -l strat
strat=$strat
for i in `seq 1 5`
do
peerannot aggregate-deep ./../music -o music_WAUM${strat}_${i} --answers ./../music/answers.json --model modelmusic --n-classes=10 --n-epochs 2000 --lr 0.001 --optimizer adam --batch-size 64 --num-workers 8 --img-size=224 -s ${strat} --scheduler -m 10000 --data-augmentation --pretrained --path-remove ./../music/identification/waum_0.05_yang/too_hard_0.05.txt
done
done

for strat in CrowdLayer[scale=0] CoNAL[scale=0]
do
echo "Strategy: ${strat}"
declare -l strat
strat=$strat
for i in `seq 1 1`
do
peerannot aggregate-deep ./../music -o music_${strat}_${i} --answers ./../music/answers.json --model modelmusic --n-classes=10 --n-epochs 2000 --lr 0.001 --optimizer adam --batch-size 64 --num-workers 8 --img-size=224 -s ${strat} --scheduler -m 10000 --pretrained --data-augmentation
done
done
