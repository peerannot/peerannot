#!/bin/bash
echo "LabelMe"

pwd

# uncomment to install if needed
# peerannot install ./../labelme/labelme.py

for strat in MV NaiveSoft DS GLAD
do
peerannot aggregate ./../labelme/ -s $strat
declare -l strat
strat=$strat
for i in `seq 1 5`
do
echo "Repetition ${i}"
peerannot train ./../labelme -o labelme_${strat}_$i -K 8 --labels=./../labelme/labels/labels_labelme_${strat}.npy --model resnet18 --n-epochs=150 --lr=0.1 --scheduler -m 50 -m 100 --scheduler --num-workers=8 --pretrained
done
done

echo "WAUMstacked"
peerannot identify ./../labelme/ -K 8 --method WAUMstacked --labels ./../labelme/answers.json --model resnet18 --n-epochs 50 --lr=0.1 --maxiter-DS=50 --alpha=0.01 --pretrained
peerannot train ./../labelme -o labelme_waum_0.01 -K 8 --labels=./../labelme/labels/labels_waumstacked_0.01.npy --model resnet18 --n-epochs=150 --lr=0.1 --scheduler -m 50 -m 100 --scheduler --num-workers=8 --path-remove ./../identification/waum_stacked_0.01_yang/too_hard_0.01.txt --pretrained


# echo "MV"
# peerannot aggregate ./../labelme/ -s MV
# peerannot train ./../labelme -o labelme_mv -K 8 --labels=./../labelme/labels/labels_labelme_mv.npy --model vgg16_bn  --n-epochs=150 --lr=0.1 --scheduler -m 50 -m 100 --scheduler --num-workers=8

# echo "Naive soft"
# peerannot aggregate ./../labelme/ -s NaiveSoft
# peerannot train ./../labelme -o labelme_naivesoft -K 8 --labels=./../labelme/labels/labels_labelme_naivesoft.npy --model vgg16_bn  --n-epochs=150 --lr=0.1 --scheduler -m 50 -m 100 --scheduler --num-workers=8

# echo "DS"
# peerannot aggregate ./../labelme/ -s DS
# peerannot train ./../labelme -o labelme_ds -K 8 --labels=./../labelme/labels/labels_labelme_ds.npy --model vgg16_bn  --n-epochs=150 --lr=0.1 --scheduler -m 50 -m 100 --scheduler --num-workers=8

# echo "GLAD"
# peerannot aggregate ./../labelme/ -s glad
# peerannot train ./../labelme -o labelme_glad -K 8 --labels=./../labelme/labels/labels_labelme_glad.npy --model vgg16_bn  --n-epochs=150 --lr=0.1 --scheduler -m 50 -m 100 --scheduler --num-workers=8

# echo "WAUMstacked"
# peerannot identify ./../labelme/ -K 8 --method WAUMstacked --labels ./../labelme/answers.json --model vgg16_bn --n-epochs 50 --lr=0.1  --maxiter-DS=50 --alpha=0.01
# peerannot train ./../labelme -o labelme_waum_0.01 -K 8 --labels=./../labelme/labels/labels_labelme_waumstacked_0.01.npy --model vgg16_bn  --n-epochs=150 --lr=0.1 --scheduler -m 50 -m 100 --scheduler --num-workers=8 --path-remove ./../identification/waum_stacked_0.01_yang/too_hard_0.01.txt



