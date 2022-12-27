Pipeline
==============

Let us consider the example of CIFAR-10H, the crowdsourced dataset of CIFAR-10.

First we need to install the dataset, meaning downloading the data and format the crowdsourced labels.

.. prompt:: bash $

   peerannot install ./datasets/cifar10H/cifar-10h.py

This creates a `train`, `val` and `test` directory with image folders ready-to-use data.
The newly created file `answers.json` contains the crowdsourced labels for each training task.

Now, we need labels to train from. Let's keep it simple and use a majority vote as example.

.. prompt:: bash $

   peerannot aggregate ./datasets/cifar10H -s MV


We have the data, and the aggregated labels. All there is left is to simply train a neural network classifier using both of them.

.. prompt:: bash $

   peerannot train ./datasets/cifar10H \
                   -o cifar10H_example -K 10 \
                   --n-epochs=10 \
                   --labels=./datasets/labels/labels_cifar-10h_mv.npy \
                   --img-size=32 \
                   --pretrained

This command trains a `resnet18` neural network (by default) with the newly aggregated labels and put the output into files beginning with `cifar10H_example`.