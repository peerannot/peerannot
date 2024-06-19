.. _cli_train:

CLI train
===============

The help documentation is available in the terminal from:

.. prompt:: bash

    peerannot train --help

All computer vision models that can be used for training are available through the ``Torchvision`` library and can be found running:

.. prompt:: bash

    peerannot modelinfo --help

.. click:: peerannot.runners.train:train
    :prog: peerannot
    :nested: full