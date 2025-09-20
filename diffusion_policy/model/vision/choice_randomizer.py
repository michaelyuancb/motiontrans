import random
from collections.abc import Sequence
import torch.nn as nn


class RandomChoice(nn.Module):
    """Apply a single transformation randomly picked from a list. This transform does not support torchscript."""

    def __init__(self, transforms, p=None):
        super().__init__()
        self.transforms = transforms
        if p is not None and not isinstance(p, Sequence):
            raise TypeError("Argument p should be a sequence")
        self.p = p

    def forward(self, *args):
        t = random.choices(self.transforms, weights=self.p)[0]
        return t(*args)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transforms={self.transforms}, p={self.p})"