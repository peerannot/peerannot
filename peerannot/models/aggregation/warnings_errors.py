class DidNotConverge(RuntimeWarning):
    def __init__(
        self,
        algorithm_name: str,
        eps: float,
        epsilon: float,
        *args,
    ) -> None:
        super().__init__(
            f"{algorithm_name} did not converge: err={eps}, {epsilon=}.",
            *args,
        )


class NotInitialized(RuntimeWarning):
    def __init__(
        self,
        algorithm_name: str,
        *args,
    ) -> None:
        super().__init__(
            f"{algorithm_name} not initialized - process at least one batch first.",
            *args,
        )


class NotNumpyArrayError(TypeError):
    def __init__(self) -> None:
        msg = "Input must be a NumPy array."
        super().__init__(msg)


class NotSliceError(TypeError):
    def __init__(self) -> None:
        msg = "Each slice must be a slice, int, or None."
        super().__init__(msg)


class TaskNotFoundError(KeyError):
    def __init__(self, task_id):
        super().__init__(f"No task found with id {task_id}")
