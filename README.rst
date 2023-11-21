*A Python library for managing and learning from crowdsourced labels in image classification tasks—*

----

|Pypi Status| |Python 3.8+| |Documentation|

The ``peerannot`` library was created to handle crowdsourced labels in classification problems.

Install
-------

To install ``peerannot``, simply run

.. code-block:: bash

    pip install peerannot

Otherwise, a ``setup.cfg`` file is located at the root directory.
Installing the library gives access to the Command Line Interface using the keyword ``peerannot`` in a bash terminal. Try it out using:

.. code-block:: bash

    peerannot --help


Quick start
---------------

Our library comes with files to download and install standard datasets from the crowdsourcing community. Those are located in the `datasets` folder

.. code-block:: bash

    peerannot install ./datasets/cifar10H/cifar10h.py

Running aggregation strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In python, we can run classical aggregation strategies from the current dataset as follows

.. code-block:: python

    for strat in ["MV", "NaiveSoft", "DS", "GLAD", "WDS"]:
        ! peerannot aggregate . -s {strat}

This will create a new folder names `labels` containing the labels in the `labels_cifar10H_${strat}.npy` file.

Training your network
^^^^^^^^^^^^^^^^^^^^^^^^^
Once the labels are available, we can train a neural network with ``PyTorch`` as follows. In a terminal:

.. code-block:: python

    for strat in ["MV", "NaiveSoft", "DS", "GLAD", "WDS"]:
        ! peerannot train . -o cifar10H_${strat} \
                    -K 10 \
                    --labels=./labels/labels_cifar-10h_${strat}.npy \
                    --model resnet18 \
                    --img-size=32 \
                    --n-epochs=1000 \
                    --lr=0.1 --scheduler -m 100 -m 250 \
                    --num-workers=8

End-to-end strategies
^^^^^^^^^^^^^^^^^^^^^^^

Finally, for the end-to-end strategies using deep learning (as CoNAL or CrowdLayer), the command line is:

.. code-block:: bash

    peerannot aggregate-deep . -o cifar10h_crowdlayer \
                         --answers ./answers.json \
                         --model resnet18 -K=10 \
                         --n-epochs 150 --lr 0.1 --optimizer sgd \
                         --batch-size 64 --num-workers 8 \
                         --img-size=32 \
                         -s crowdlayer

For CoNAL, the hyperparameter scaling can be provided as ``-s CoNAL[scale=1e-4]``.


Peerannot and the crowdsourcing formatting
----------------------------------------------

In ``peerannot``, one of our goals is to make crowdsourced datasets under the same format so that it is easy to switch from one learning or aggregation strategy without having to code once again the algorithms for each dataset.

So, what is a crowdsourced dataset? We define each dataset as:

.. code-block:: bash

    dataset
    ├── train
    │     ├── class1
    │     ├── ...
    │     ├── data as imagename-<key>.png
    │     └── ...
    ├── val
    ├── test
    ├── dataset.py
    ├── metadata.json
    └── answers.json


The crowdsourced labels for each training task are contained in the ``anwers.json`` file. They are formatted as follows:

.. code-block:: bash

    {
        0: {<worker_id>: <label>, <another_worker_id>: <label>},
        1: {<yet_another_worker_id>: <label>,}
    }

Note that the task index in the ``answers.json`` file might not match the order of tasks in the ``train`` folder... Thence, each task's name contains the associated votes file index.
The number of tasks in the ``train`` folder must match the number of entry keys in the ``answers.json`` file.

The ``metadata.json`` file contains general information about the dataset. A minimal example would be:

.. code-block:: bash

    {
        "name": <dataset>,
        "n_classes": K,
        "n_workers": <n_workers>,
    }


Create you own dataset
^^^^^^^^^^^^^^^^^^^^^^^

The ``dataset.py`` is not mandatory but is here to facilitate the dataset's installation procedure. A minimal example:

.. code-block:: python

    class mydataset:
        def __init__(self):
            self.DIR = Path(__file__).parent.resolve()
            # download the data needed
            # ...

        def setfolders(self):
            print(f"Loading data folders at {self.DIR}")
            train_path = self.DIR / "train"
            test_path = self.DIR / "test"
            valid_path = self.DIR / "val"

            # Create train/val/test tasks with matching index
            # ...

            print("Created:")
            for set, path in zip(
                ("train", "val", "test"), [train_path, valid_path, test_path]
            ):
                print(f"- {set}: {path}")
            self.get_crowd_labels()
            print(f"Train crowd labels are in {self.DIR / 'answers.json'}")

        def get_crowd_labels(self):
            # create answers.json dictionnary in presented format
            # ...
            with open(self.DIR / "answers.json", "w") as answ:
                json.dump(dictionnary, answ, ensure_ascii=False, indent=3)


.. |Pypi Status| image:: https://github.com/peerannot/peerannot/actions/workflows/python-publish.yml/badge.svg?branch=main
   :target: https://github.com/peerannot/peerannot/actions/workflows/python-publish.yml
.. |Python 3.8+| image:: https://github.com/peerannot/peerannot/actions/workflows/pytest.yml/badge.svg
   :target: https://github.com/peerannot/peerannot/actions/workflows/pytest.yml
.. |Documentation| image:: https://github.com/peerannot/peerannot.github.io/actions/workflows/deploy-jekyll.yml/badge.svg
   :target: https://peerannot.github.io
