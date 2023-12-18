# %%
import json
import sys
from pathlib import Path
import os
import numpy as np

# %%
if len(sys.argv) != 3:
    sourcePath = (
        "/home/dubar/Bureau/peerannot/fork/peerannot/datasets/labelme/data/LabelMe/"
    )
    targetPath = (
        "/home/dubar/Bureau/peerannot/fork/peerannot/datasets/labelme/Testsyslink/"
    )
    answersPath = "/home/dubar/Bureau/peerannot/fork/peerannot/datasets/labelme/data/LabelMe/answers.txt"
    filenameTrainPath = "/home/dubar/Bureau/peerannot/fork/peerannot/datasets/labelme/data/LabelMe/filenames_train.txt"
else:
    sourcePath = sys.argv[1]
    if not os.path.exists(sourcePath):
        print("source Directory not existing")

    targetPath = sys.argv[2]
    if not os.path.exists(targetPath):
        print("target Directory not existing")

# %%


def verifyFolders():
    print("Coucou")
    if os.path.exists(sourcePath + "/train"):
        trainDir = sourcePath + "/train"
        print("train Directory exists")
    else:
        return False

    if os.path.exists(sourcePath + "/test"):
        testDir = sourcePath + "/test"
        print("test Directory already exists")
    else:
        return False

    hasValid = False
    if os.path.exists(sourcePath + "/valid"):
        hasValid = True
        validDir = sourcePath + "/valid"
        print("valid Directory already exists")
    return True


###########################################################################


def createFolder():
    if not os.path.exists(targetPath + "/train"):
        os.makedirs(targetPath + "/train")
    if not os.path.exists(targetPath + "/test"):
        os.makedirs(targetPath + "/test")
    if not os.path.exists(targetPath + "/valid"):
        os.makedirs(targetPath + "/valid")


def SyslinkCopy():
    for i, labels in enumerate(os.listdir(sourcePath + "train/")):
        if not os.path.exists(targetPath + labels):
            os.makedirs(targetPath + "train/" + labels)
        for j, file in enumerate(os.listdir(sourcePath + "train/" + labels)):
            os.symlink(
                sourcePath + "train/" + labels + "/" + file,
                targetPath + "train/" + labels + "/" + file,
            )

    for i, file in enumerate(os.listdir(testDir)):
        os.symlink(sourcePath + "/train/" + file, targetPath + "/test/" + file)

    for i, file in enumerate(os.listdir(validDir)):
        os.symlink(sourcePath + "/train/" + file, targetPath + "/valid/" + file)


verifyFolders()
createFolder()
SyslinkCopy()
# %%

dirTarget = Path(targetPath).resolve()
dirSource = Path(sourcePath).resolve()

crowdlabels = np.loadtxt(answersPath)
orig_name = np.loadtxt(filenameTrainPath, dtype=str)

res_train = {task: {} for task in range(crowdlabels.shape[0])}
for id_, task in enumerate(crowdlabels):
    where = np.where(task != -1)[0]
    for worker in where:
        res_train[id_][int(worker)] = int(task[worker])

for i, labels in enumerate(os.listdir(sourcePath + "train/")):
    if not os.path.exists(targetPath + "train/" + labels):
        os.makedirs(targetPath + "train/" + labels)
for j, file in enumerate((dirSource / "train").glob("*/*")):
    print(file)
    parent = file.parent.name
    currentFile = (
        dirTarget
        / "train"
        / parent
        / f"{file.stem}-{np.where(orig_name == file.name)[0][0]}.jpg"
    )
    print("Current file :", file)
    # os.symlink(
    #     file.absolute(),
    #     dirTarget
    #     / "train"
    #     / parent
    #     / f"{file.stem}-{np.where(orig_name == file.name)[0][0]}.jpg",
    # )


with open(
    "/home/dubar/Bureau/peerannot/fork/peerannot/datasets/labelme/Testsyslink/answers.json",
    "w",
) as answ:
    json.dump(res_train, answ, ensure_ascii=False, indent=3)
# %%
