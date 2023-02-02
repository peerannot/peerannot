#!/bin/bash
echo "CIFAR-10H"

pwd

# uncomment to install if needed
# peerannot install ./../cifar10H/cifar10h.py

for strat in MV NaiveSoft DS GLAD
do
echo "Strategy: ${strat}"
peerannot aggregate ./../cifar10H/ -s $strat
declare -l strat
strat=$strat
for i in `seq 1 5`
do
echo "Repetition ${i}"
peerannot train ./../cifar10H -o cifar10H_${strat}_$i -K 10 --labels=./../cifar10H/labels/labels_cifar-10h_${strat}.npy --model resnet18 --img-size=32 --n-epochs=150 --lr=0.1 --scheduler -m 50 -m 100 --scheduler --num-workers=8
done
done

for strat in WDS
do
echo "Strategy: ${strat}"
peerannot aggregate ./../cifar10H/ -s $strat
declare -l strat
strat=$strat
for i in `seq 1 5`
do
echo "Repetition ${i}"
peerannot train ./../cifar10H -o cifar10H_${strat}_$i -K 10 --labels=./../cifar10H/labels/labels_cifar-10h_${strat}.npy --model resnet18 --img-size=32 --n-epochs=1000 --lr=0.1 --scheduler -m 100 -m 250 --num-workers=8
done
done

echo "Strategy: WAUM"
peerannot identify ./../cifar10H/ -K 10 --method WAUM --labels ./../cifar10H/answers.json --model resnet18 --n-epochs 50 --lr=0.1 --img-size=32 --maxiter-DS=50 --alpha=0.01
for i in `seq 1 5`
do
echo "Repetition ${i}"
peerannot train ./../cifar10H -o cifar10H_waum_0.01 -K 10 --labels=./../cifar10H/labels/labels_waum_0.01.npy --model resnet18 --img-size=32 --n-epochs=150 --lr=0.1 -m 50 -m 100 --scheduler --num-workers=8 --path-remove ./../cifar10H/identification/waum__0.01_yang/too_hard_0.01.txt
done

for strat in MV NaiveSoft DS GLAD
do
echo "Strategy: WAUM  + ${strat}"
declare -l strat
strat=$strat
for i in `seq 1 5`
do
echo "Repetition ${i}"
peerannot train ./../cifar10H -o cifar10H_waum_0.01_${strat}_${i} -K 10 --labels=./../cifar10H/labels/labels_cifar-10h_${strat}.npy --model resnet18 --img-size=32 --n-epochs=150 --lr=0.1 -m 50 -m 100 --scheduler --num-workers=8 --path-remove ./../identification/waum__0.01_yang/too_hard_0.01.txt
done
done

