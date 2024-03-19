import click
import json
import sys
from pathlib import Path
import os
import numpy as np
from datetime import datetime


class CustomDataset:
    def __init__(self):
        self.DIR = Path(__file__).parent.resolve()
        self.val_ratio = 0.2

    ### Functions to compute answers.json from different sources
    def computeJsonAnswers(self, answersPath, outputname):
        with open(answersPath, "r") as f:
            answers = json.load(f)
        with open(
            outputname,
            "w",
        ) as answ:
            json.dump(answers, answ, ensure_ascii=False, indent=3)

    def computeInvertJsonAnswers(self, answersPath, outputname):
        with open(answersPath, "r") as f:
            answers = json.load(f)

        max_task_id = 0
        for worker in answers:
            for task in answers[worker]:
                max_task_id = max(max_task_id, int(task))
        taskAnswers = {task: {} for task in range(max_task_id + 1)}
        for worker in answers:
            for task in answers[worker]:
                taskAnswers[int(task)][int(worker)] = answers[worker][task]

        with open(
            outputname,
            "w",
        ) as answ:
            json.dump(taskAnswers, answ, ensure_ascii=False, indent=3)

    def computeRodriguesAnswers(self, answersPath, outputname):
        crowdlabels = np.loadtxt(answersPath)
        res_train = {task: {} for task in range(crowdlabels.shape[0])}
        for id_, task in enumerate(crowdlabels):
            where = np.where(task != -1)[0]
            for worker in where:
                res_train[id_][int(worker)] = int(task[worker])
        with open(
            outputname,
            "w",
        ) as answ:
            json.dump(res_train, answ, ensure_ascii=False, indent=3)

    ############################################################################

    def writeSymlink(self, folder, file, currentPath, position):
        parent = file.parent.name
        currentFile = currentPath / folder / parent / f"{file.stem}-{position}.jpg"
        reversedParents = (currentFile.parents)[1::-1]

        for subfolder in reversedParents:
            os.makedirs(subfolder, exist_ok=True)
        if currentFile.exists():
            return
        os.symlink(
            file.absolute(),
            currentPath / folder / parent / f"{file.stem}-{position}.jpg",
        )

    def numberWorkers(self, answers_path):
        # We compute how many workers there are in the answers.json file
        with open(answers_path, "r") as f:
            answers = json.load(f)
        answersList = [list(x) for x in answers.values()]
        workersSet = sorted(set(int(x) for xs in answersList for x in xs))
        return workersSet

    # Main function, called by the executable
    def setfolders(
        self,
        no_task,
        answers_format,
        answers,
        metadata,
        label_names,
        files_path,
        train_path,
        test_ground_truth_format,
        test_ground_truth,
        test_path,
        val_path,
    ):
        # Compute the answers.json file from different sources
        answersPath = Path(answers)
        if answers_format == 0:
            self.computeRodriguesAnswers(answersPath, "answers.json")
        elif answers_format == 1:
            self.computeJsonAnswers(answersPath, "answers.json")
        elif answers_format == 2:
            self.computeInvertJsonAnswers(answersPath, "answers.json")

        ##################################################################
        ## From now on, "./answers.json" is complete                    ##
        ##################################################################

        if no_task:
            return

        hasTest = True
        if test_path == "":
            hasTest = False

        hasLabel = True
        if label_names == "":
            hasLabel = False

        # Create train and val folders and load files to copy the task data
        os.makedirs("train/", exist_ok=True)
        os.makedirs("val/", exist_ok=True)

        filenameTrainPath = Path(files_path)
        orig_name = np.loadtxt(filenameTrainPath, dtype=str)

        # Create symlinks in the train, val and test folders
        currentPath = Path(".")
        trainPath = Path(train_path)

        taskMaxId = 0
        if val_path == "":
            click.echo(
                "No val path provided, samples from the train set will be used instead"
            )
            random_seed = 12345
            rng = np.random.default_rng(random_seed)
            for j, file in enumerate(trainPath.glob("*/*")):
                rand_val = rng.random()
                file_destination = "train"
                if rand_val < self.val_ratio:
                    file_destination = "val"

                taskId = np.where(orig_name == file.name)[0][0]
                taskMaxId = max(taskId, taskMaxId)
                self.writeSymlink(
                    file_destination,
                    file,
                    currentPath,
                    taskId,
                )
        else:
            file_destination = "train"
            for j, file in enumerate(trainPath.glob("*/*")):
                taskId = np.where(orig_name == file.name)[0][0]
                taskMaxId = max(taskId, taskMaxId)
                self.writeSymlink(
                    file_destination,
                    file,
                    currentPath,
                    taskId,
                )

            valPath = Path(val_path)
            file_destination = "val"
            taskMaxId += 1
            for j, file in enumerate(valPath.glob("*/*")):
                self.writeSymlink(
                    file_destination,
                    file,
                    currentPath,
                    taskMaxId,
                )
                taskMaxId += 1

        if hasTest:
            testPath = Path(test_path)
            os.makedirs("test/", exist_ok=True)

            taskMaxId += 1
            for j, file in enumerate(testPath.glob("*/*")):
                self.writeSymlink("test", file, currentPath, taskMaxId)
                taskMaxId += 1

        ########################################################################
        # From now on the Train, Val and Test folders are complete            ##
        ########################################################################
        if hasLabel:
            labelNamesPath = Path(label_names)
            label_namesTab = np.loadtxt(labelNamesPath, dtype=str)
        else:
            trainPath = Path("./train/")
            label_namesTab = next(os.walk(trainPath))[1]
        # Create the test_groundTruth.json file
        if hasTest:
            if test_ground_truth == "":
                testPath = Path("./test/")
                res_test = {task: {} for task in range(orig_name.shape[0], taskMaxId)}
                for j, file in enumerate(testPath.glob("*/*")):
                    parent = file.parent.name
                    parentId = np.where(label_namesTab == parent)[0][0]
                    taskId = file.stem.split("-")[-1]
                    res_test[int(taskId)][0] = int(parentId)
                with open(
                    "./test_groundTruth.json",
                    "w",
                ) as answ:
                    json.dump(res_test, answ, ensure_ascii=False, indent=3)
            else:
                if test_ground_truth_format == 0:
                    self.computeRodriguesAnswers(
                        test_ground_truth, "test_groundTruth.json"
                    )
                elif test_ground_truth_format == 1:
                    self.computeJsonAnswers(test_ground_truth, "test_groundTruth.json")
                elif test_ground_truth_format == 2:
                    self.computeJsonAnswers(test_ground_truth, "test_groundTruth.json")

        # Verify that the metadata.json file exists or is complete. Create it if not
        if metadata == "":
            metadata = {
                "name": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                "n_classes": len(label_namesTab),
                "n_workers": self.numberWorkers(Path("./answers.json"))[-1] + 1,
                "nb_workers": len(self.numberWorkers(Path("./answers.json"))),
            }
            with open("./metadata.json", "w") as answ:
                json.dump(metadata, answ, ensure_ascii=False, indent=3)
        else:
            with open(metadata, "r") as f:
                metadata = json.load(f)

            if not "name" in metadata.keys():
                metadata["name"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

            if not "n_classes" in metadata.keys():
                metadata["n_classes"] = len(label_namesTab)

            if not "n_workers" in metadata.keys():
                metadata["n_workers"] = (
                    self.numberWorkers(Path("./answers.json"))[-1] + 1
                )

            if not "nb_workers" in metadata.keys():
                metadata["nb_workers"] = len(self.numberWorkers(Path("./answers.json")))

            with open("./metadata.json", "w") as answ:
                json.dump(metadata, answ, ensure_ascii=False, indent=3)
