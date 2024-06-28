.. _cli_datasets:

CLI datasets
===============

To install a dataset (from the example datasets or your custom dataset) use the command:

.. prompt:: bash

    peerannot install installationFile.py

For either case, a Python file describing the installation will be needed.

Example datasets
----------------

To install an example library, only the installation file is needed. For example
want to install the ``cifar10H`` library, run:

.. prompt:: bash

    cd datasets/cifar10H & peerannot install cifar10h.py


Custom datasets
---------------

To install your dataset, you will have to use the ``customDataset.py`` installation file located at `datasets/`
and add multiple arguments depending on the structure of your dataset.

Only the ``answers`` file and the ``answers-format`` arguments must always be included.
The ``answers-format`` is an integer option that represents how your answers file is encoded. for now here are the format supported:

* ``0``: A text file that represents the answers in a :math:`|task| * |worker|` matrix where each entry is either the id value of a label or -1 in case of the absence of answer. For the toy-dataset of peerannot, we have the following matrix answers file (3 tasks, 4 workers):

.. centered::
    :math:`\begin{bmatrix}
    1 & 0 & 1 & 1\\ 
    1 & -1 & 0 & 0\\ 
    -1 & 1 & -1 & 1
    \end{bmatrix}`
    

* ``1``, The main format of peerannot. A JSON file where the main keys are the tasks' id. Each task has a list with the worker's id as a key and the answer as the value. listed by id. For the toy-dataset of peerannot, we have the following matrix answers file (3 tasks, 4 workers):

.. code-block:: json

    {
        "0": {
            "0": 1,
            "1": 0,
            "2": 1,
            "3": 1
        },
        "1": {
            "0": 1,
            "2": 0,
            "3": 0
        },
        "2": {
            "1": 1,
            "3": 1
        }
    }

* ``2``, Similar to the precedent JSON format except the worker and the task order is reversed. 
A JSON file where the main keys are the worker's ids. Each worker has a list with the task's id as a key and the answer as the value. For the toy-dataset of peerannot, we have the following matrix answers file (3 tasks, 4 workers):

.. code-block:: json

    {
        "0": {
            "0": 1,
            "1": 1
        },
        "1": {
            "0": 0,
            "2": 1
        },
        "2": {
            "0": 1,
            "1": 0
        },
        "3": {
            "0": 1,
            "1": 0,
            "2": 1
        }
    }

Taskless dataset
^^^^^^^^^^^^^^^^

If your dataset has no task, then you can add the ``no-task`` flag with the ``answers`` and ``answers-format`` argument.

.. prompt:: bash

    cd datasets/MyDataset & peerannot install ../customDataset.py --no-task \\
    --answers answersFile.json --answers-format 1

Dataset with tasks
^^^^^^^^^^^^^^^^^^

In the case your dataset has tasks (if you want to train a model for image classification). 
A ``train-set`` must be included and you will have to specify its path.
A ``files-path`` also has to be given. It should include the path to the file with the same
order as the one in the ``answers`` file.

A validation set can be provided with the ``val-set`` option but is not mandatory. In case
a validation set is not provided it will be created with 20% of the train set.

Finally, label names can be provided in a file with the option ``label-names`` which can help
construct the structure of the dataset (especially if the test set has no ground truth file).
In case it's not given, it will be assumed that the structure of the dataset is similar to a Pytorch
ImageFolder dataset (see https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
where tasks are arranged inside folders per labels.

Here are some examples of commands to create custom datasets:

Creation of a dataset with no task:

.. prompt:: bash

    peerannot install datasets/customDataset.py --answers-format 2 \\
    --answers PATH_TO_ANSWERS_FILE/answers.json --no-task

Creation of a dataset with a train, val and test set:

.. prompt:: bash

    peerannot install datasets/customDataset.py --train-path PATH_TO_TRAIN_DIR \\
    --test-path PATH_TO_TEST_DIR --val-path PATH_TO_VAL_DIR \\
    --answers PATH_TO_ANSWERS_FILE/answers.txt \\
    --files-path PATH_TO_FILENAMES_FILE/filenames.txt \\
    --label-names PATH_TO_LABELNAMES_FILE/labelNames.txt

Creation of a dataset with only a train set:
    
.. prompt:: bash

    peerannot install datasets/customDataset.py --train-path PATH_TO_TRAIN_DIR \\
    --answers-format 1 --files-path PATH_TO_FILENAME_FILE/filenames.txt \\
    --answers PATH_TO_ANSWERS_FILE/answers.json \\
    --label-names PATH_TO_LABELNAMES_FILE/labelNames.txt

The help documentation is available in the terminal from:

.. prompt:: bash

    peerannot install --help


.. click:: peerannot.runners.datasets:install
    :prog: peerannot
    :nested: full
