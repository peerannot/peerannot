.. _add_dataset:

Add a dataset to peerannot
=====================================

This tutorial shows how to add a new dataset in peerannot.

.. Hint::
    If not yet done, please go to :ref:`get started page <get_started>` to install the peerannot library.


What is a dataset?
-------------------------

Datasets are located in the `datasets/` directory.
You can create a new dataset by creating a new directory in `datasets/` and adding the following files:

- `mydataset.py` containing how to install the dataset
- `metadata.json` containing all relevant information about the dataset
- `answers.json` a `.json` file containing the answers to the questions asked in the crowdsourced experiment

If the task images are also available, they can be installed in the `mydataset.py` file using the `setfolders` method.

.. code-block:: python
    :caption: datasets/mydataset/mydataset.py

    import json
    from pathlib import Path

    class MyDataset:
        name = 'mydataset'

        def __init__(self):
            self.DIR = Path(__file__).parent.resolve()
            ...  # download all necessary files

        def setfolders(self):
            ...  # split data into train/val/test folders
            selg.get_crowd_labels()

        def get_crowd_labels(self):
            ...  # save the crowd labels in a .json file
            with open(self.DIR / "answers.json", "w") as answ:
                json.dump(mylabels, answ, ensure_ascii=False, indent=3)


The dataset folder should look like this:

.. code-block:: bash

    mydataset/
    ├── mydataset.py     # install file
    ├── train/
    │   ├── ...          # existing images with crowd labels
    ├── val/
    │   ├── ...          # existing images for validation
    ├── test/
    │   ├── ...          # existing images with known labels
    ├── answers.json     # crowdsourced answers
    └── metadata.json    # dataset information

