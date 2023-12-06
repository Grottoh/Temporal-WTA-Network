from typing import Optional
import torch
import numpy as np
from enum import Enum
import math

from .spike_generator import SpikeGenerator
from ..utils.util import get_seed
from ..data_loaders.data import Data

class SGStatic(SpikeGenerator):
    """ Converts static non-binary data samples to spikes. """

    # Determine degree to which neurons fire in response to specific amplitude:
    # Threshold: spike prob at max when <X_i> exceeds threshold, zero otherwise
    THRESHOLD = "THRESHOLD"
    # Linear: spike probability increases linearly as <X_i> increases
    LINEAR = "LINEAR"

    def __init__(
            self,
            hz_spike: int,
            hz_timestep: int = 1000,
            d_seconds: float = 150/1000,
            threshold: float = 0,
            inverted: bool = False,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        # Determine the firing probability of active neurons at each timestep
        self.hz_timestep = hz_timestep # How many timesteps per second

        # How many spikes per second (on average) at maximum amplitude
        self.hz_spike = hz_spike 

        # The spike probability of each active neuron at each timestep
        self.p_spike = self.hz_spike / self.hz_timestep

        # Set the stimulus duration in seconds and timesteps respectively
        self.d_seconds = d_seconds
        self._n_timesteps = math.ceil(d_seconds * self.hz_timestep)

        # Set threshold if and only if <mode> is <THRESHOLD>
        if self.mode == self.THRESHOLD:
            self.threshold = threshold

        # By default high values of <X_i> are associated with high spiking
        # probability; if inverted it is the other way around
        self.inverted = inverted
    
    def set_shape(self):
        """ Determine the shape of the to be generated spikes. """
        self.n_timesteps = self._n_timesteps
        self.shape = (self.n_timesteps,) + self.layer.data.shape
    
    def generate_spikes(self, X_i: torch.FloatTensor) -> torch.FloatTensor:
        """ Convert the given data sample to spikes. """

        # Ensure data points are within the range [0, 1]
        if torch.min(X_i) < 0.0 or torch.max(X_i) > 1.0:
            raise ValueError(
                f"<{self.__name__}> requires that input" +
                f" tensor X_i is normalized to be in the range [0, 1].")
        
        # Linear: spike probability increases linearly as <X_i> increases
        if self.mode == self.LINEAR:
            intensity = X_i.clone()
            if self.inverted: # If enabled, invert the spiking probabilities
                intensity = 1 - intensity
        
        # Threshold: spike probability at the maximum when <X_i> exceeds
        # the given threshold, zero otherwise
        elif self.mode == self.THRESHOLD:
            intensity = X_i > self.threshold
            if self.inverted: # If enabled, invert the spiking probabilities
                intensity = ~intensity
        
        # Unknown: raise an error
        else:
            raise ValueError(f"SGStatic mode <{self.mode}> is not known.")
        
        # Weigh the intensity with maximum spiking probability <self.p_spike>
        p_spike = self.p_spike * intensity
        
        # Determine at random (according to <p_spike>) which neurons spike
        randomness = torch.rand(
            self.shape, 
            generator=self.rng, 
            device=self.device,
            )
        S_i = randomness < p_spike
        S_i = S_i.to(torch.float32)
        
        return S_i