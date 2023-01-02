#!/bin/bash
echo "LabelMe"

pwd

# uncomment to install if needed
# peerannot install ./../labelme/labelme.py

for strat in MV NaiveSoft DS GLAD
do
echo "Strategy: ${strat}"
peerannot aggregate ./../labelme/ -s $strat
declare -l strat
strat=$strat
for i in `seq 1 5`
do
echo "Repetition ${i}"
peerannot train ./../labelme -o labelme_${strat}_$i -K 8 --labels=./../labelme/labels/labels_labelme_${strat}.npy --model modellabelme --n-epochs=1100 --lr=0.005 --scheduler -m 750 -m 1000 --num-workers=8 --pretrained --img-size=224 --data-augmentation --batch-size=228 --optimizer=adam
done
done

echo "Strategy: WAUMstacked"
peerannot identify ./../labelme/ -K 8 --method WAUMstacked --labels ./../labelme/answers.json --model modellabel --n-epochs 500 --lr=0.005 --maxiter-DS=50 --alpha=0.01 --pretrained --optimizer=adam --batch-size=228
for i in `seq 1 5`
do
echo "Repetition ${i}"
peerannot train ./../labelme -o labelme_waum_0.01 -K 8 --labels=./../labelme/labels/labels_waumstacked_0.01.npy --model modellabelme --n-epochs=1100 --lr=0.005 --scheduler -m 750 -m 1000 --num-workers=8 --path-remove ./../identification/waum_stacked_0.01_yang/too_hard_0.01.txt --pretrained --data-augmentation --optimizer=adam --batch-size=228
done

for strat in MV NaiveSoft DS GLAD
do
echo "Strategy: WAUM stacked + ${strat}"
declare -l strat
strat=$strat
for i in `seq 1 5`
do
echo "Repetition ${i}"
peerannot train ./../labelme -o labelme_waum_0.01_${strat}_${i} -K 8 --labels=./../labelme/labels/labels_waumstacked_0.01.npy --model modellabelme --n-epochs=1100 --lr=0.005 --scheduler -m 750 -m 1000 --num-workers=8 --path-remove ./../identification/waum_stacked_0.01_yang/too_hard_0.01.txt --pretrained --data-augmentation --optimizer=adam --batch-size=228
done
done

for strat in CoNAL[scale=0] CoNAL[scale=1e-4] CrowdLayer[scale=0] CrowdLayer[scale=1e-4]
do
echo "Strategy: ${strat}"
declare -l strat
strat=$strat
for i in `seq 1 5`
do
peerannot aggregate-deep ./../labelme -o labelme_${strat}_${i} --answers ./../labelme/answers.json --model modellabelme --n-classes=8 --n-epochs 1100 --lr 0.005 --optimizer adam --batch-size 228 --num-workers 8 --img-size=224 -s ${strat} --scheduler -m 750 -m 1000 --data-augmentation --pretrained
done
done
