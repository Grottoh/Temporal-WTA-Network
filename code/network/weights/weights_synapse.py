from typing import Callable, Optional, Any
import torch
import numpy as np
from contextlib import suppress

import network.utils.util as util
from .weights import Weights
from ..layers.layer import Layer
from ..synapses.synapse import Synapse

class WeightsSynapse(Weights):
    
    W_MIN = 0.0
    W_MAX = 1.0

    def __init__(
            self,
            hz_timestep: int,
            hz_layer: int,
            cst_curve: float = 1.0,
            w_min: None = None,
            w_max: None = None,
            **kwargs
            ) -> None:
        super().__init__(w_min=self.W_MIN, w_max=self.W_MAX, **kwargs)

        # Determine which timescale weight updates should adhere to
        self.hz_timestep = hz_timestep
        self.hz_layer = hz_layer
        self.iv_spike = self.hz_timestep / self.hz_layer

        # Higher value means exponential increase in <_dw> as <w> gets larger
        # NOTE: if cst_curve > 1 then weight updates of low weights may explode
        self.cst_curve = cst_curve
    
    def generate_weights(self, **kwargs) -> None:
        super().generate_weights(**kwargs)

        # Keeps track for each synapse, how many timesteps ago each other
        # synapse received a spike relative to said synapse
        self.trace_timing = torch.full(
            size=self.synapse.shape_a+self.synapse.shape_a,
            fill_value=self.iv_spike,
            dtype=torch.float32,
            device=self.device,
            )
        
        # The amount by which to update the synapse weights of a spiking neuron
        self.trace_strength = torch.zeros(
            size=self.synapse.shape_a+self.synapse.shape_a,
            dtype=torch.float32,
            device=self.device,
            )

    @property
    def synapse(self) -> Synapse:
        """ Alternative reference to host component, which is a synapse. """
        return self.host

    @property
    def layer_b(self) -> Layer:
        """ The post-synaptic layer. """
        return self.synapse.layer
    
    @property
    def saveables(self) -> dict[str, Any]:
        """<__dict__> exluding attributes that can be saved, but shouldn't. """
        forgettables = [] # Specifies the to be forgotten attributes
        with suppress(AttributeError): forgettables.append(self.trace_timing)
        with suppress(AttributeError): forgettables.append(self.trace_strength)
        return self._saveables(forgettables=forgettables)
        
    def learn(self) -> None:
        """ Trace pre-synaptic spike activity and update the weights following
        a post-synaptic spike. """
        super().learn()
            
        def trace_curve(trace_timing_ka):
            """Go 1 to 0 at exponential rate as <trace_timing_ka> increases."""
            return (self.iv_spike-trace_timing_ka)/self.iv_spike
        
        self.trace_timing = torch.clip(self.trace_timing+1, max=self.iv_spike)
        
        # Update the strength trace of all synapses receiving spikes
        for k_a in util.where(self.synapse.spikes):
            self.trace_strength[k_a] += trace_curve(self.trace_timing[k_a])

        # Update the timing trace of all synapses receiving spikes
        for k_a in util.where(self.synapse.spikes):

            # Trace going into gate k_a is reset
            self.trace_timing[k_a] = self.iv_spike

        for k_a in util.where(self.synapse.spikes):
            # Trace going out of k_a is set to max
            self.trace_timing[util.full_slice(self.synapse.shape_a)+k_a] = 0
            
        for k_b in self.synapse.layer.ks:
            for _ in range(self.eta_star): # Repeat <eta_star> times

                # Retrieve the synapse weights of spiking neuron <k_b>
                weights_kb = self.weights[k_b].clone()

                # Determine strength of weight update according to the weights
                _dw = torch.exp( -(weights_kb - self.w_max) * self.cst_curve)
                
                # Weigh the update with the synapse trace
                dw = ( self.trace_strength*_dw - 1 ) / np.exp(self.cst_curve)

                # Apply the weight update (weighed by learning rate <eta>)
                weights_kb = weights_kb + self.eta[k_b] * dw
                    
                # Ensure the weights are within bounds
                weights_kb = torch.clip(
                    weights_kb,
                    min=self.w_min,
                    max=self.w_max,
                    )
                    
                # Update the weights tensor
                self.weights[k_b] = weights_kb
                    
                # Update the learning rate of neuron <k_b>
                self.eta[k_b] /= self.eta[k_b]**(1/self.eta_decay) + 1
                
                # NOTE: eta-hack_v1
                # if self.eta[k_b] < 1.0: --- bad performance (29.75%)
                #     self.eta[k_b] /= self.eta[k_b]**(1/self.eta_decay) + 1
                # else:
                #     self.eta[k_b] /= self.eta[k_b]**(-self.eta_decay) + 1
                
                # # NOTE: eta-hack_v2 --- bad performance (34.20%)
                # if self.eta[k_b] < 1.0:
                #     self.eta[k_b] /= self.eta[k_b]**(1/self.eta_decay) + 1
                # else:
                #     self.eta[k_b] /= self.eta[k_b]**(-1/self.eta_decay) + 1
            
        if self.synapse.layer.spiked:
            self.trace_timing = torch.full_like(
                self.trace_timing, 
                fill_value=self.iv_spike,
                )
            self.trace_strength = torch.zeros_like(self.trace_strength)
    
    def next(self):
        """ Called upon transitioning to a new stimulus. """
        super().next()
        
        self.trace_timing = torch.full_like(
            self.trace_timing, 
            fill_value=self.iv_spike,
            )
        self.trace_strength = torch.zeros_like(self.trace_strength)
