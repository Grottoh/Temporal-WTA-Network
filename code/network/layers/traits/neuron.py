from dataclasses import dataclass
import torch

import network.utils.util as util
from ...component import Trait
from ..layer import Layer

class Neuron(Trait):
    
    @property
    def layer(self) -> Layer:
        """ Alternative reference to host component, which is a WTA layer. """
        return self.host

class NeuronSoftmax(Neuron):

    def __init__(
            self, 
            hz_timestep: int, 
            hz_layer: int, 
            en_variable_duration: bool = False,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        self.hz_timestep = hz_timestep
        self.hz_layer = hz_layer
        self.iv_spike = self.hz_timestep / self.hz_layer

        # NOTE: hack - allow a spike to be triggered when the stimulus is done
        self.en_variable_duration = en_variable_duration
    
    def step(self) -> None:
        """ The spiking probability of each neuron is the softmax function of
        their membrane potentials. """
        
        # If interval <iv_spike> has passed, produce one or multiple spikes
        if (self.layer.t + 1) % round(self.iv_spike) == 0:
            
            # Mark that a neuron has spiked
            spiked = True
            
            # Determine the spiking probability of each neuron
            
            # The <-1> ensures probability zero for <mp==0>
            if torch.max(self.layer.mps)>0 and torch.max(self.layer.mps) < 25:
                p_spikes = ((torch.exp(self.layer.mps) - 1) / 
                            (torch.exp(self.layer.mps) - 1).sum())
            # When there are high membrane potentials the exponent will explode
            # unless e.g. the maximum is subtracted; at this point 0 values
            # will not yield exactly zero, but will be so small that they don't
            # matter anymore anyway
            else:
                p_spikes = (torch.exp(
                    self.layer.mps-torch.max(self.layer.mps)) / 
                    torch.exp(self.layer.mps-torch.max(self.layer.mps)).sum())
            
            # Generate up to <n_max_simultaneous_spikes> spikes
            _ks = torch.multinomial(
                p_spikes.flatten(), 
                num_samples=self.layer.n_max_simultaneous_spikes, 
                replacement=True, 
                generator=self.layer.rng,
                ).to(dtype=torch.int32)
            ks = [self.layer.KS[_k] for _k in _ks]
            spikes = torch.zeros_like(self.layer.spikes)
            spikes[ list( zip( *ks ) ) ] = 1
        
        else: # Do not spike
            spiked = False # Mark that no neuron has spiked
            ks = []
            spikes = torch.zeros_like(self.layer.spikes)
        
        # Assign the spike details to the layer
        self.layer.spikes = spikes
        self.layer.ks = ks
        self.layer.spiked = spiked
    
    def force_spike(self):

        # Mark that a neuron has spiked
        spiked = True
            
        # Determine the spiking probability of each neuron
        
        # The <-1> ensures probability zero for <mp==0>
        if torch.max(self.layer.mps)>0 and torch.max(self.layer.mps) < 25:
            p_spikes = ((torch.exp(self.layer.mps) - 1) / 
                        (torch.exp(self.layer.mps) - 1).sum())
        # When there are high membrane potentials the exponent will explode
        # unless e.g. the maximum is subtracted; at this point 0 values
        # will not yield exactly zero, but will be so small that they don't
        # matter anymore anyway
        else:
            p_spikes = (torch.exp(
                self.layer.mps-torch.max(self.layer.mps)) / 
                torch.exp(self.layer.mps-torch.max(self.layer.mps)).sum())
        
        # Generate up to <n_max_simultaneous_spikes> spikes
        _ks = torch.multinomial(
            p_spikes.flatten(), 
            num_samples=self.layer.n_max_simultaneous_spikes, 
            replacement=True, 
            generator=self.layer.rng,
            ).to(dtype=torch.int32)
        ks = [self.layer.KS[_k] for _k in _ks]
        spikes = torch.zeros_like(self.layer.spikes)
        spikes[ list( zip( *ks ) ) ] = 1
        
        # Assign the spike details to the layer
        self.layer.spikes = spikes
        self.layer.ks = ks
        self.layer.spiked = spiked

class NeuronStochastic(Neuron):

    def __init__(self, cst_p_spike:float=20.0, **kwargs):
        super().__init__(**kwargs)

        self.cst_p_spike = cst_p_spike
    
    def step(self) -> None:
        """ The spiking probability of each neuron increases exponentially as
        its membrane potential increases. """
        
        # Determine which neurons spike according to their firing probability
        # p_spike = torch.exp((self.layer.mps-self.layer.mps_max) * 
        #                     self.layer.mps_max)
        p_spike = torch.exp(
            (self.layer.mps-self.layer.mps_max)/self.layer.mps_max *
            self.cst_p_spike
            )
        randomness = torch.rand(
            self.layer.shape, 
            generator=self.layer.rng, 
            device=self.layer.device,
            )
        spikes = randomness < p_spike
        ks = util.where(spikes)
        n_spikes = len(ks)
        spiked = n_spikes > 0

        if spiked:
            breakpoint

        if (n_spikes > self.layer.n_max_simultaneous_spikes):
            
            # Select at most <n_max_simultaneous_spikes> neurons to spike
            _ks = torch.multinomial(
                torch.ones(size=(n_spikes,), device=self.layer.device),
                num_samples=min(n_spikes, 
                                self.layer.n_max_simultaneous_spikes),
                replacement=False,
                generator=self.layer.rng
                ).to(dtype=torch.int32)
            ks = [ks[_k] for _k in _ks]
            spikes = torch.zeros_like(self.layer.spikes)
            spikes[ list( zip( *ks ) ) ] = 1
            n_spikes = self.layer.n_max_simultaneous_spikes
                
        else: # Get the indices of the neurons that spiked
            ks = util.where(spikes)
            spikes = spikes.to(torch.float32)
        
        # Assign the spike details to the self.layer
        self.layer.spikes = spikes
        self.layer.ks = ks
        self.layer.spiked = spiked