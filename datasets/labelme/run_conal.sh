#!/bin/bash
echo "LabelMe CoNAL"

pwd

# uncomment to install if needed
# peerannot install ./../labelme/labelme.py

peerannot aggregate-deep ./../labelme -o labelme_conal --answers ./../labelme/answers.json -s conal[scale=1e-5] \
    --model modellabelme --img-size=224 --pretrained --n-classes=8 \
    --n-epochs=1100 --lr=0.005 -m 1000 -m 1050 --scheduler \
    --batch-size=228 --optimizer=adam --num-workers=8 --data-augmentation
