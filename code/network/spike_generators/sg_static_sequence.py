from __future__ import annotations
from typing import Optional
import torch
import numpy as np
from enum import Enum
import math

from .sg_static import SGStatic
from ..utils.util import get_seed
from ..data_loaders.data import Data

class SGStaticSequence(SGStatic):

    def __init__(
            self,
            dim_time: int,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        # Determine which dimension should become the time dimension
        self.dim_time = dim_time
    
    @classmethod
    def instant(
        cls,
        dim_time: int,
        **kwargs
        ) -> SGStaticSequence:
        """ Each segment of the sequence is represented in a single timestep,
        where there is exactly one spike per non-zero value. """
        return cls(
            dim_time=dim_time,
            mode=cls.THRESHOLD,
            hz_spike=1000,
            hz_timestep=1000,
            d_seconds=1/1000,
            threshold=0,
            **kwargs
            )
    
    def set_shape(self):
        """ Determine the shape of the to be generated spikes. """

        # Convert shape to list so it can be altered
        shape = list(self.layer.data.shape)

        # Remove time dimension from shape and determine its size
        self.n_timesteps = self._n_timesteps * shape.pop(self.dim_time)

        # Set time dimension as the first dimension
        self.shape = (self.n_timesteps,) + tuple(shape)
    
    def generate_spikes(self, X_i: torch.FloatTensor) -> torch.FloatTensor:
        """ Convert the given data sample to spikes. """

        # Set the time dimension to be the first dimension
        _X_i = X_i.swapaxes(self.dim_time, 0)

        # Stretch the time dimension
        _X_i = _X_i.repeat_interleave(self._n_timesteps, dim=0)

        # Generate spikes
        S_i = super().generate_spikes(X_i=_X_i)

        return S_i