# Welcome to Peerannot

The `peerannot` library was created to handle crowdsourced labels in classification problems.

## Quick start

Let us consider how we would work with, for example, the `cifar10H` dataset.
We assume that we are in the `cifar10H` directory containing the `cifar10H.py` file.

First, install the dataset with `peerannot install ./cifar10h.py`.
Then, we can try classical label aggregation strategies as follows:

```bash
for strat in MV NaiveSoft DS GLAD WDS
do
echo "Strategy: ${strat}"
peerannot aggregate . -s $strat
done
```

This will create a new folder names `labels` containing the labels in the `labels_cifar10H_${strat}.npy` file.
Once we have the labels, we can train a neural network with `pytorch` as follows:

```bash
for strat in MV NaiveSoft DS GLAD WDS
do
echo "Strategy: ${strat}"
declare -l strat
strat=$strat
peerannot train . -o cifar10H_${strat} \
                  -K 10 \
                  --labels=./labels/labels_cifar-10h_${strat}.npy
                  --model resnet18 \
                  --img-size=32 \
                  --n-epochs=1000 \
                  --lr=0.1 --scheduler -m 100 -m 250 \
                  --num-workers=8
done
```

As the `WAUM` purpose is to identify ambiguous tasks, the command to run the identification is:

```bash
peerannot identify . -K 10 --method WAUM \
                     --labels ./answers.json \
                     --model resnet18 --n-epochs 50 \
                     --lr=0.1 --img-size=32 \
                     --maxiter-DS=50 \
                     --alpha=0.01
```

Then, one can train on the pruned dataset with any aggregation strategy as follows:
```bash
# run WAUM + DS strategy
peerannot train . -o cifar10H_waum_0.01_DS \
                -K 10 \
                --labels= ./labels/labels_cifar-10h_ds.npy \
                --model resnet18 --img-size=32 \
                --n-epochs=150 --lr=0.1 -m 50 -m 100 --scheduler \
                --num-workers=8 \
                --path-remove ./identification/waum_0.01_yang/too_hard_0.01.txt
```

Finally, for the end-to-end strategies using deep learning (as CoNAL or CrowdLayer), the command line is:

```bash
peerannot aggregate-deep . -o cifar10h_crowdlayer \
                         --answers ./answers.json \
                         --model resnet18 -K=10 \
                         --n-epochs 150 --lr 0.1 --optimizer sgd \
                         --batch-size 64 --num-workers 8 \
                         --img-size=32 \
                         -s crowdlayer
```

For CoNAL, the hyperparameter scaling can be provided as `-s CoNAL[scale=1e-4]`.

## Peerannot and crowdsourcing formatting

In `peerannot`, one of our goal is to make crowdsourced datasets under the same format so that it is easy to switch from one learning or aggregation strategy without having to code once again the algorithms for each dataset.

So, what is a crowdsourced dataset? We define each dataset as:

```bash
dataset
    ├── train
    │     ├── class0
    │     │     ├─ task0-<vote_index_0>.png
    │     │     ├─ task1-<vote_index_1>.png
    │     │     ├─ ...
    │     │     └─ taskn0-<vote_index_n0>.png
    │     ├── class1
    │     ├── ...
    │     └── classK
    ├── val
    ├── test
    ├── dataset.py
    ├── metadata.json
    └── answers.json
```

The crowdsourced labels for each training task are contained in the `anwers.json` file. They are formatted as follows:

```json
{
    0: {<worker_id>: <label>, <another_worker_id>: <label>},
    1: {<yet_another_worker_id>: <label>,}
}
```

Note that because the task index in the `answers.json` file might not match the order of tasks in the `train` folder, each task has in its name the associated index in the votes file.
The number of tasks in the `train` folder **must** match the number of entry keys into the `answers.json` file.

The `metadata.json` file contains general information about the dataset. A minimal example would be:
```json
{
    "name": <dataset>,
    "n_classes": K,
    "n_workers": <n_workers>
}
```
The `dataset.py` is not mandatory, but is here to facilitate the dataset's installation procedure. A minimal example is:
```python
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
```

