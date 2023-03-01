import numpy as np
from tqdm import tqdm
from pathlib import Path


def with_confusion(
    n_worker, true_labels, K, matrix, rng, *, difficulties=None, **kwargs
):
    """Generate an answer from confusion matrices.

    :param n_worker: number of workers
    :type n_worker: int
    :param true_labels: list of true label for each task
    :type true_labels: list of int between 0 and K-1
    :param K: number of classes
    :type K: int
    :param matrix: Confusion matrix for each worker
    :type matrix: array of shape (n_worker, K, K)
    :param rng: Numpy generator
    :type rng: generator
    :return: answers for each task
    :rtype: dictionnary of length n_task
    """

    n_task = len(true_labels)
    answers = {int(i): {} for i in range(n_task)}
    workerload = kwargs.get("workerload", n_task)
    feedback = kwargs.get("feedback", n_worker)
    loads = np.array([0] * n_worker)
    feeds = np.array([0] * n_task)
    for i in tqdm(range(n_task), total=n_task, desc="Task"):
        y_istar = true_labels[i]
        ans = {}
        possible_workers = np.where(loads < workerload)[0]
        # print(i, loads, feeds)
        howmany = min(min(workerload, len(possible_workers)), feedback)
        if howmany == 0:
            raise ValueError(
                f"Could not satisfy conditions:\n \t n_task={n_task}, \n\t workerload = {workerload}, \n\tfeedback effort={feedback}\n\t n_worker={n_worker}."
            )
        if kwargs["imbalance_votes"]:
            howmany = rng.choice(range(howmany)) + 1  # range: (0: howmany -1)
        feeds[i] += howmany
        workers = rng.choice(
            possible_workers,
            replace=False,
            size=howmany,
        )
        if difficulties is None:
            for j in workers:
                ans[int(j)] = int(rng.choice(K, p=matrix[j][y_istar]))
        else:
            for j in workers:
                if difficulties[i] == "easy":
                    ans[int(j)] = int(y_istar)
                elif difficulties[i] == "random":
                    ans[int(j)] = int(rng.choice(K))
                else:  # difficulty == "hard"
                    ans[int(j)] = int(rng.choice(K, p=matrix[j][y_istar]))
        loads[workers] += 1
        answers[int(i)] = dict(sorted(ans.items()))
    return answers


def hammer_spammer(n_worker, true_labels, K, rng, **kwargs):
    """Simulate a case with hammers (no confusion) and spammers (answer at random)

    :param n_worker: number of workers
    :type n_worker: int
    :param true_labels: true label for each task
    :type true_labels: list of int between 0 and K-1
    :param K: number of classes
    :type K: int
    :param ratio: Ratio of spamers amongst workers (n_spammers / n_workers)
    :type ratio: float in (0,1)
    :param rng: Numpy generator
    :type rng: generator
    """
    ratio = kwargs.get("ratio")
    n_spammers = int(ratio * n_worker)
    n_hammers = n_worker - n_spammers
    matrices = [np.eye(K)] * n_hammers
    matrices.extend([1 / K * np.ones((K, K))] * n_spammers)
    matrices = np.stack(matrices)
    answers = with_confusion(n_worker, true_labels, K, matrices, rng, **kwargs)
    np.save(Path(kwargs["folder"]) / "matrices.npy", matrices)
    return answers


def student_teacher(n_worker, true_labels, K, rng, **kwargs):
    """Correlated mistakes with students following teachers

    :param n_worker: number of workers in total
    :type n_worker: int
    :param true_labels: true label for each task
    :type true_labels: list of int between 0 and K-1
    :param K: number of classes
    :type K: int
    :param ratio: Ratio of students amongst workers (n_student / n_workers)
    :type ratio: float in (0,1)
    :param rng: Numpy generator
    :type rng: generator
    """
    ratio = kwargs.get("ratio")
    n_student = int(n_worker * ratio)
    n_teacher = n_worker - n_student
    print(f"Simulating {n_student} students and {n_teacher} teachers")
    if kwargs.get("matrix_file", None):
        matrices = np.load(kwargs["matrix_file"])[:n_teacher, :, :]
    else:
        alpha = [1] * K
        matrices = np.stack(
            [rng.dirichlet(alpha, size=K) for _ in range(n_teacher)]
        )  # (n_teacher, K, K)
    # assign each student to one teacher
    following = rng.choice(n_teacher, replace=True, size=n_student)
    matrices = np.vstack([matrices, matrices[following, :, :]])
    answers = with_confusion(n_worker, true_labels, K, matrices, rng, **kwargs)
    np.save(Path(kwargs["folder"]) / "matrices.npy", matrices)
    np.save(
        Path(kwargs["folder"]) / "student_teacher_association.npy",
        np.hstack(
            (
                n_teacher + np.arange(n_student).reshape(-1, 1),
                following.reshape(-1, 1),
            )
        ),
    )
    return answers


def matrix_independent(n_worker, true_labels, K, rng, **kwargs):
    """Answer following a confusion matrix.
    Keywords arguments may contain the path to the worker confusion matrices, the workerload or the feedback.
    - workerload: Number of tasks answered by workers
    - Feedback: Number of answers per tasks

    :param n_worker: number of workers
    :type n_worker: int
    :param true_labels: true label for each task
    :type true_labels: list of int between 0 and K-1
    :param K: number of classes
    :type K: int
    :param rng: Random generator
    :type rng: numpy generator
    :return: answers in the peerannot format {task: {worker: label}}
    :rtype: dictionnary of length n_task
    """
    if kwargs.get("matrix_file", None):
        matrices = np.load(kwargs["matrix_file"])
    else:
        alpha = [1] * K
        matrices = np.stack(
            [rng.dirichlet(alpha, size=K) for _ in range(n_worker)]
        )  # (n_worker, K, K)
    answers = with_confusion(n_worker, true_labels, K, matrices, rng, **kwargs)
    np.save(Path(kwargs["folder"]) / "matrices.npy", matrices)
    return answers


def discrete_difficuty(n_worker, true_labels, K, rng, **kwargs):
    """Answer following a confusion matrix and the task difficulty.
    Keywords arguments may contain the path to the worker confusion matrices, the workerload, the feedback or if a `random` difficulty level is added.
    - workerload: Number of tasks answered by workers
    - feedback: Number of answers per tasks
    - random (float in (0,1)): Creates a level of difficulty `random` in addition to `easy` and `hard` that represents tasks impossible to label correctly. The probability for a task to be random is controlled by this parameter: 0 means no random tasks, 1 means all tasks are random.

    :param n_worker: number of workers
    :type n_worker: int
    :param true_labels: true label for each task
    :type true_labels: list of int between 0 and K-1
    :param K: number of classes
    :type K: int
    :param rng: Random generator
    :type rng: numpy generator
    :return: answers in the peerannot format {task: {worker: label}}
    :rtype: dictionnary of length n_task
    """
    ratio = kwargs.get("ratio")  # ratio of good workers amongst the bad
    ratio_diff = kwargs.get("ratio_diff", 1)  # easy tasks amongst hard
    good_workers = int(ratio * n_worker)
    bad_workers = n_worker - good_workers
    p_random = kwargs.get("random", 0)
    p_hard = (1 - p_random) / (ratio_diff + 1)
    nt = len(true_labels)
    difficulties = rng.choice(
        ["easy", "hard", "random"],
        p=[ratio_diff * p_hard, p_hard, p_random],
        size=nt,
    )
    if kwargs.get("matrix_file", None):
        matrices = np.load(kwargs["matrix_file"])
        assert matrices.shape == (
            n_worker,
            K,
            K,
        ), f"With difficulty, the matrix should be of shape {(n_worker, K, K)} and not {matrices.shape}"
    else:
        good_matrices = []
        arr = np.arange(K)
        for _ in range(good_workers):  # build matrices for good workers
            mat = rng.dirichlet([0.2] * K, size=K)
            argmax_ = np.argmax(mat, axis=1)
            mat[arr, arr], mat[arr, argmax_] = mat[arr, argmax_], mat[arr, arr]
            good_matrices.append(mat)
        matrix_good = np.stack(good_matrices)  # (good_worker, K, K)
        matrix_bad = np.stack(
            [rng.dirichlet([1] * K, size=K) for _ in range(bad_workers)]
        )  # (bad_worker, K, K)
        matrices = np.vstack((matrix_good, matrix_bad))
    answers = with_confusion(
        n_worker,
        true_labels,
        K,
        matrices,
        rng,
        difficulties=difficulties,
        **kwargs,
    )
    np.save(Path(kwargs["folder"]) / "matrices.npy", matrices)
    np.save(Path(kwargs["folder"]) / "difficulties.npy", difficulties)
    return answers


simulation_strategies = {
    "hammer-spammer": hammer_spammer,
    "student-teacher": student_teacher,
    "independent-confusion": matrix_independent,
    "discrete-difficulty": discrete_difficuty,
}
