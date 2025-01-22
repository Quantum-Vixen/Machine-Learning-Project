from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TrainingAlgorithm import TrainingAlgorithm

class StoppingCondition:
    def __init__(self):
        self.alg: TrainingAlgorithm = None
    
    def set_alg(self, alg: TrainingAlgorithm) -> None:
        self.alg = alg

    @property
    def is_satisfied(self) -> bool:
        pass

class ThresholdOnTrainingError(StoppingCondition):
    """
    Parameters
    ----------
    threshold: float
    patience: int
    """
    def __init__(self, threshold: float, patience: int):
        super().__init__()
        self.threshold: float = threshold
        self.patience: int = patience
    
    @property
    def is_satisfied(self) -> bool:
        current_training_error: float = self.alg.current_tr_err
        if current_training_error < self.threshold:
            self.consecutive_epochs += 1
            return self.consecutive_epochs > self.patience
        else:
            self.consecutive_epochs = 0
            return False
    
    def __str__(self):
        return f"TR Err threshold: {self.threshold}"

class TrainingErrorPlateau(StoppingCondition):
    """
    This condition stops the training if the training error doesn't decrease anymore.
    The training is stopped if for patience consecutive epochs, the training error doesn't
    decrease more than threshold_fraction * training_error.
    """
    def __init__(self, threshold_fraction: float, patience: int):
        super().__init__()
        self.threshold: float = threshold_fraction
        self.patience: int = patience

        self.old_tr_err: float = float('inf')

    @property
    def is_satisfied(self) -> bool:
        current_training_error: float = self.alg.current_tr_err
        if current_training_error > (1 - self.threshold) * self.old_tr_err:
            self.consecutive_epochs += 1
            self.old_tr_err = current_training_error
            return self.consecutive_epochs > self.patience
        else:
            self.consecutive_epochs = 0
            self.old_tr_err = current_training_error
            return False
    
    def __str__(self):
        return f"TR Err plateau: {self.threshold * 100}%"