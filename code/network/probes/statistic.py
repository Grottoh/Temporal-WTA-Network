from __future__ import annotations
from typing import Optional, Any
from dataclasses import dataclass
import torch
from collections import deque
import numpy as np

from network.weights.weights import Weights

from ..layers.layer import Layer
from ..layers.layer_sensory import LayerSensory

@dataclass
class Statistic:

    value: Optional[Any] = None # Determined by child class
    device: str = 'cpu' # Device to use when dealing with torch tensors
    
    @staticmethod
    def select(stat_a: Statistic, stat_b: Statistic) -> Statistic:
        """ Return one of two same-typed statistics based on their
        parameters. Base Probe does not have a preference, but child classes
        may. For example, one statistic may be identical except for one
        having a larger buffer size, thus superseding the other. """

        # Given statistics must be of the same type
        if not type(stat_a) == type(stat_b):
            raise TypeError(f"Given statistics <{stat_a}> and <{stat_b}>" +
                              " must be of the same type.")
        
        # Any actual preference might be indicated by child class
        return stat_a 
    
    def on_run(self):
        """ Called when starting to a new run. Necessary because some
        statistics require data to be loaded before being able to initialize
        properly. """
        pass # Implemented by child class
    
    def on_start(self):
        """ Called when starting to process a new stimulus. """
        pass # Implemented by child class

    def step(self) -> None:
        """ Called at the end of a timestep. """
        pass # Implemented by child class

    def on_end(self) -> None:
        """ Called when at the end of processing a specific stimulus. """
        pass # Implemented by child class

class SpikesSinceStimulus(Statistic):
    """ Keeps track of a layer's spikes since the onset of a new stimulus. """

    def __init__(
            self,
            layer: Layer,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
        self.layer = layer # The count is with respect to this layer
    
    @property
    def key(self):
        """ Key is equal if the statistics concern the same objects. """
        return type(self).__name__ + repr(self.layer)
    
    def on_run(self):
        """ At onset start of a run, initialize the value to keep track of. """
        self.value = torch.zeros(
            size=self.layer.shape, 
            dtype=torch.int16,
            device=self.device,
            )

    def on_start(self):
        """ Reset each neuron's spike count at the start of a new stimulus. """
        self.value = torch.zeros_like(self.value)
    
    def step(self):
        """ Increment each neuron's spike count at each timestep. """
        self.value += self.layer.spikes.to(torch.bool)

class SpikeTimes(Statistic):
    """ Keeps track of exact spike times of a layer's neurons since the onset
    of a new stimulus. """

    def __init__(
            self,
            layer: Layer,
            n_timesteps: int,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
        self.layer = layer # The count is with respect to this layer
        self.n_timesteps = n_timesteps # The amount of timesteps per stimulus
    
    @property
    def key(self):
        """ Key is equal if the statistics concern the same objects. """
        return type(self).__name__ + repr(self.layer)
    
    def on_run(self):
        """ At onset start of a run, initialize the value to keep track of. """
        self.value = torch.zeros(
            size=(self.n_timesteps,) + self.layer.shape,
            dtype=torch.bool,
            device=self.device,
            )
        self.t = 0

    def on_start(self):
        """ Reset spike times at the start of a new stimulus. """
        self.value = torch.zeros_like(self.value)
        self.t = 0
    
    def step(self):
        """ Increment each neuron's spike count at each timestep. """
        self.value[self.t] += self.layer.spikes.to(torch.bool)
        self.t += 1

class SpikesHistory(Statistic):
    """ Keeps track of a layer's history of spikes up to a given number of past
    stimuli. """

    def __init__(
            self,
            layer: Layer,
            n_past: int,
            spikes_since_stimulus: SpikesSinceStimulus,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        self.layer = layer # The spike counts are with respect to this layer
        self.n_past = n_past # Maximum number of past stimuli to keep track of
        self._spikes_since_stimulus = spikes_since_stimulus # Sub-count

        # Ensure that the sub-count concerns the same layer
        assert self.layer == self._spikes_since_stimulus.layer

    @staticmethod
    def select(stat_a: SpikesHistory, stat_b: SpikesHistory) -> SpikesHistory:
        """ Return one of two same-typed statistics based on their parameters.
        For example, one statistic may be identical except for one having a
        larger buffer size, thus superseding the other. """

        # Given statistics must be of type SpikesHistory
        if not type(stat_a) == type(stat_b) == SpikesHistory:
            raise TypeError(f"Given statistics <{stat_a}> and <{stat_b}>" +
                              " must be of type <SpikesHistory>")
        
        # Return the SpikesHistory that keeps track of a larger history
        return stat_a if stat_a.n_past >= stat_b.n_past else stat_b
    
    @property
    def key(self):
        """ Key is equal if the statistics concern the same objects. """
        return type(self).__name__ + repr(self.layer)
    
    @property
    def spikes_since_stimulus(self) -> torch.ShortTensor:
        """ Return the present value of <_spikes_since_stimulus>. """
        return self._spikes_since_stimulus.value
    
    def on_run(self):
        """ At onset start of a run, initialize the value to keep track of. """
        self.value = torch.zeros(
            size=(self.n_past,) + self.layer.shape,
            dtype=torch.float16, # Must be float for <torch.mean()> and such
            device=self.device,
            )
        self.index = 0
    
    def on_end(self):
        """ At end of stimulus, remember its spikes counts. """
        self.value[self.index%self.n_past] *= 0 # TODO: more efficient in another way?
        self.value[self.index%self.n_past] += self.spikes_since_stimulus
        self.index += 1
    
    def get(self, m_past: Optional[int] = None) -> torch.HalfTensor:
        """ Get the last <m_past> spike counts"""
        # TODO: identical to <get> of CountStimulusNeuron, remove repetition

        # By default retrieve <self.n_past> spike counts
        m_past = self.n_past if m_past == None else m_past

        # Ensure that we do not try to retrieve more than can be stored
        if m_past > self.n_past:
            raise ValueError(f"Given <m_past={m_past} may not be larger" +
                             f" than <self.n_past={self.n_past}>.")
        
        # Determine the indices of the past <m_past> spike counts
        indices = np.arange(
            start=self.index - m_past, 
            stop =self.index, 
            dtype=np.int32,
            ) % self.n_past
        
        return self.value[indices]


class CountNeuronLabel(Statistic):
    """ Counts the number of times each neuron spiked for each label. """

    def __init__(
            self,
            layer: Layer,
            layer_sensory: LayerSensory, 
            spikes_since_stimulus: SpikesSinceStimulus,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        self.layer = layer # The count is with respect to this layer
        self.layer_sensory = layer_sensory # Contains label information
        self._spikes_since_stimulus = spikes_since_stimulus # Sub-count

        # Ensure that the sub-count concerns the same layer
        assert self.layer == self._spikes_since_stimulus.layer
    
    @property
    def key(self):
        """ Key is equal if the statistics concern the same objects. """
        return (type(self).__name__ + 
                repr(self.layer) + 
                repr(self.layer_sensory) +
                self._spikes_since_stimulus.key)
    
    @property
    def spikes_since_stimulus(self) -> torch.ShortTensor:
        """ Return the present value of <_spikes_since_stimulus>. """
        return self._spikes_since_stimulus.value
    
    def on_run(self):
        """ At onset start of a run, initialize the value to keep track of. """
        self.value = torch.zeros(
            size=self.layer.shape + (self.layer_sensory.data.n_classes,),
            dtype=torch.int32,
            device=self.device,
            )
    
    def on_end(self):
        """ At end of stimulus, increment spike counts for stimulus label. """
        self.value[..., self.layer_sensory.T_i] += self.spikes_since_stimulus

# TODO: see if I create a parent class for this and e.g. SpikesHistory (same
# behaviour with <n_past>)
class CountStimulusNeuron(Statistic):
    """ For each stimulus, counts the number of times each neuron spiked. """

    def __init__(
            self,
            layer: Layer,
            n_past: int,
            spikes_since_stimulus: SpikesSinceStimulus,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        self.layer = layer # The count is with respect to this layer
        self.n_past = n_past # Max amount of stimuli for which to count
        self._spikes_since_stimulus = spikes_since_stimulus # Sub-count

        # Ensure that the sub-count concerns the same layer
        assert self.layer == self._spikes_since_stimulus.layer

    @staticmethod
    def select(
        stat_a: CountStimulusNeuron, 
        stat_b: CountStimulusNeuron
        ) -> CountStimulusNeuron:
        """ Return one of two same-typed statistics based on their parameters.
        For example, one statistic may be identical except for one having a
        larger buffer size, thus superseding the other. """

        # Given statistics must be of type CountStimulusNeuron
        if not type(stat_a) == type(stat_b) == CountStimulusNeuron:
            raise TypeError(f"Given statistics <{stat_a}> and <{stat_b}>" +
                              " must be of type <CountStimulusNeuron>")
        
        # Return the CountStimulusNeuron that keeps track of a larger history
        return stat_a if stat_a.n_past >= stat_b.n_past else stat_b
    
    @property
    def key(self):
        """ Key is equal if the statistics concern the same objects. """
        return (type(self).__name__ +
                repr(self.layer) +
                self._spikes_since_stimulus.key)
    
    @property
    def spikes_since_stimulus(self) -> torch.ShortTensor:
        """ Return the present value of <_spikes_since_stimulus>. """
        return self._spikes_since_stimulus.value
    
    def on_run(self):
        """ At onset start of a run, initialize the value to keep track of. """
        self.value = torch.zeros(
            size=(self.n_past,)+self.layer.shape,
            dtype=torch.int16,
            device=self.device,
            )
        self.index = 0
    
    def on_end(self):
        """ At end of stimulus, increment spike counts for stimulus label. """
        self.value[self.index%self.n_past] = self.spikes_since_stimulus
        self.index += 1
    
    def get(self, m_past: Optional[int] = None) -> torch.HalfTensor:
        """ Get the last <m_past> stimulus-neuron spike counts"""
        # TODO: identical to <get> of SpikesHistory, remove repetition

        # By default retrieve <self.n_past> spike counts
        m_past = self.n_past if m_past == None else m_past

        # Ensure that we do not try to retrieve more than can be stored
        if m_past > self.n_past:
            raise ValueError(f"Given <m_past={m_past} may not be larger" +
                             f" than <self.n_past={self.n_past}>.")
        
        # Determine indices of the past <m_past> stimulus-neuron spike counts
        indices = np.arange(
            start=self.index - m_past, 
            stop =self.index, 
            dtype=np.int32,
            ) % self.n_past
        
        return self.value[indices]

# class SizeWeights(Statistic):
#     """ Keeps track of the number of weights that lie at or below the given
#     thresholds. """

#     def __init__(
#             self,
#             weights: Weights,
#             thresholds: list[float],
#             **kwargs
#             ) -> None:
#         super().__init__(**kwargs)
#         self.weights = weights # The count is with respect to these weights
#         self.thresholds = thresholds
#         self.n_weights = torch.numel(self.weights.weights)
#         breakpoint
    
#     @property
#     def key(self):
#         """ Key is equal if the statistics concern the same objects. """
#         return type(self).__name__ + repr(self.weights) + str(self.thresholds)
    
#     def on_run(self):
#         """ At onset start of a run, initialize the value to keep track of. """
#         self.value = dict( 
#             [ (threshold, []) for threshold in self.thresholds ] 
#             )
    
#     def on_end(self):
#         """ Count for each threshold the number of weights that are lesser
#         than or equal to it. """
#         for threshold, ls_n_less_or_equal in self.value.items():
#             ls_n_less_or_equal.append(
#                 torch.sum( self.weights.weights <= threshold )
#             )