from typing import Optional, Union, Any
import torch
import numpy as np
from contextlib import suppress

import network.utils.util as util
from .layer import Layer
from ..data_loaders.data import Data
from ..spike_generators.spike_generator import SpikeGenerator

class LayerSensory(Layer):

    def __init__(
            self,
            shape: Union[ tuple[int, ...], list[int] ] = tuple(),
            **kwargs
            ) -> None:
        # NOTE: shape is alternatively set with <set_traits(...)>
        super().__init__(shape=shape, **kwargs)
    
    def set_traits(
            self,
            data: Data,
            spike_generator: SpikeGenerator,
            ) -> None:
        """ 
        Set traits of the layer.

        Traits are not optional, because the shape of the spike-generator 
        depends on the data, and, in turn, the shape of the layer depends
        on the shape of the spike-generator.
        """
        
        # Set the data trait, determining which data is being observed
        if issubclass(type(data), Data): # Set specific data class
            self.data = data
        else: # Given value is invalid (non-data) type
            raise TypeError(
                f"Trait <data> must be of type <{Data}>," +
                f" but <{type(data)}> was given."
                )
        
        # Set spike-gen trait, determining how data is translated into spikes
        if issubclass(type(spike_generator), SpikeGenerator): # Set class
            self.spike_generator = spike_generator # Set the spike generator

            # Set the shape of the spike generator according to the layer data
            self.spike_generator.set_shape()
            
            self._set_shape(shape=self.spike_generator.shape[1:])

        else: # Given value is invalid (non-spike-generator) type
            raise TypeError(
                f"Trait <spike_generator> must be type <{SpikeGenerator}>," +
                f" but <{type(spike_generator)}> was given."
                )
    
    @property
    def saveables(self) -> dict[str, Any]:
        """<__dict__> exluding attributes that can be saved, but shouldn't. """
        forgettables = [] # Specifies the to be forgotten attributes
        with suppress(AttributeError): forgettables.append(self.S_i)
        # TODO: move below forgettables to super class (also for LayerWTA)
        with suppress(AttributeError): forgettables.append(self.spikes)
        with suppress(AttributeError): forgettables.append(self.ks)
        with suppress(AttributeError): forgettables.append(self.spiked)
        return self._saveables(forgettables=forgettables)
    
    @property
    def X_i(self) -> torch.FloatTensor:
        """ The current data sample. """
        return self.data.X[self.index]
    
    @property
    def T_i(self) -> np.ndarray[int]:
        """ The class of the current data sample. """
        return self.data.T[self.index]
    
    def load_data(self, **kwargs) -> None:
        """ Load either train or test data and prepare the first stimulus. """

        # Load a subset of the data
        self.data.load_data(**kwargs)

        # Start iterating over the data at data sample zero
        self.index = 0
                
        # Generate spikes according to <X_i>
        self.S_i = self.spike_generator.generate_spikes(self.X_i)

        # NOTE: hack
        if hasattr(self.data, 'variable_durations'):
            self.variable_duration = self.data.variable_durations[self.index]
    
    def step(self) -> None:
        super().step()
        self.spikes = self.S_i[self.t]
        self.ks = util.where(self.spikes)
        self.spiked = torch.any(self.spikes)

    def next(self) -> None:
        """ Move on to the next stimulus and convert it to spikes. """
        super().next()
        self.index = (self.index + 1) % self.data.n_samples
        self.S_i = self.spike_generator.generate_spikes(self.X_i)
        
        # NOTE: hack
        if hasattr(self.data, 'variable_durations'):
            self.variable_duration = self.data.variable_durations[self.index]