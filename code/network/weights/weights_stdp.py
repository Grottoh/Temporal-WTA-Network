from contextlib import suppress
from typing import Any, Callable, Optional
import torch
import numpy as np

from .weights import Weights
from ..layers.layer import Layer
from ..axons.axon import Axon

# TODO: Come up with more elegant STDP dynamics? E.g. one that does not need
# <.clip>, and perhaps has some interesting/descriptive mathematical
# foundation. Perhaps for <eta> (learning rate) as well.
class WeightsStdp(Weights):
    """ Weights that evolve according to a local STDP rule. """

    W_MAX = 1.0

    def __init__(
            self,
            hz_timestep: int,
            hz_layer: int,
            decay_slow: float = 0.99,
            decay_fast: float = 0.50,
            cst_curve: float = 1.0,
            cst_beta: float = 0.0,
            w_max: None = None,
            **kwargs
            ) -> None:
        super().__init__(w_max=self.W_MAX, **kwargs)
        
        # Determine which timescale weight updates should adhere to
        self.hz_timestep = hz_timestep
        self.hz_layer = hz_layer
        self.iv_spike = self.hz_timestep / self.hz_layer

        if self.hz_layer > 1000:
            raise ValueError(
                f"Number of spikes per second (<{self.hz_layer}>) cannot" +
                 " exceed the number of timesteps per second" +
                f" (<{self.hz_timestep}>)."
                )
        
        # Higher value means slower decay; should be in range [0, 1]
        self.decay_slow = decay_slow
        self.decay_fast = decay_fast

        # Higher value means exponential increase in <_dw> as <w> gets larger
        # NOTE: if cst_curve > 1 then weight updates of low weights may explode
        self.cst_curve = cst_curve

        # Weights must be in the range [<=0, 1]
        if self.w_min > 0:
            raise ValueError(f"Weight minumum <w_min={self.w_min}> must be" +
                              " smaller or equal to zero.")

        # TODO: visualize weight update for various weights
        
        # NOTE: won't do anything if 0.0; at >0.0 it will increase the
        # short-term learning rate of a neuron in response to spikes
        self.cst_beta = cst_beta
    
    def generate_weights(self, **kwargs) -> None:
        super().generate_weights(**kwargs)

        # Initialize the trace of each axon, tracing pre-synaptic activity
        self.trace = torch.zeros(
            size=self.axon.shape_a,
            dtype=torch.float32,
            device=self.device,
            )
        
        # Decay slowly until <t_till_fast==0>
        self.t_till_fast = torch.zeros(
            size=self.axon.shape_a,
            dtype=torch.float32,
            device=self.device,
            )
        
        # NOTE
        self.beta = torch.ones_like(self.eta)

        self.weights = torch.ones_like(self.weights) # NOTE: all weight 1
    
    # @property
    # def saveables(self) -> dict[str, Any]:
    #     """<__dict__> exluding attributes that can be saved, but shouldn't. """
    #     forgettables = [] # Specifies the to be forgotten attributes
    #     with suppress(AttributeError): forgettables.append(self.beta)
    #     with suppress(AttributeError): forgettables.append(self.t_till_fast)
    #     with suppress(AttributeError): forgettables.append(self.trace)
    #     return self._saveables(forgettables=forgettables)

    @property
    def axon(self) -> Axon:
        """ Alternative reference to the host component, which is an axon. """
        return self.host

    @property
    def layer_b(self) -> Layer:
        """ The post-synaptic layer. """
        return self.axon.synapse.layer
        
    def learn(self) -> None:
        """ Trace pre-synaptic spike activity and update the weights following
        a post-synaptic spike. """
        super().learn()
        #return # NOTE: have weights remain unchanged

        self.t_till_fast = torch.clip(self.t_till_fast-1, min=0)
        self.trace[self.t_till_fast>0] *= self.decay_slow
        self.trace[self.t_till_fast==0] *= self.decay_fast

        self.trace[self.axon.spikes==1] = 1.0
        self.t_till_fast[self.axon.spikes==1] = self.iv_spike
        
        if self.layer_b.spiked: # If a neuron in the post-synaptic layer spiked
            for k_b in self.layer_b.ks: # Iterate over spiking neurons <ks_b>
                for _ in range(self.eta_star): # Repeat <eta_star> times
                    
                    # Retrieve the weights of spiking neuron <k_b>
                    weights_kb = self.weights[k_b].clone()
                    
                    # Negative weights are updated as if their value is zero
                    idc_negative = weights_kb < 0
                    weights_kb[idc_negative] = 0

                    # Determine strength of weight update according to weights
                    # _dw = np.exp(-weights_kb * self.cst_curve)

                    # Normalize <_dw> such that at <W_MAX==1.0>, <_dw> is 1.0;
                    # this ensures that at <W_MAX>, <dw> is equal to zero
                    # _dw = _dw / np.exp(-self.W_MAX*self.cst_curve)

                    # NOTE: should be equivalent to the above
                    # Determine strength of weight update according to weights
                    _dw = torch.exp(-(weights_kb-self.W_MAX) * self.cst_curve)

                    # Weigh the update with the pre-synaptic trace
                    dw = self.trace*(_dw) - 1 
                    
                    # Restrict weight update <dw>
                    dw = dw / np.exp(self.cst_curve)

                    # Apply the weight update (weighed by learning rate <eta>)
                    #weights_kb = weights_kb + self.eta[k_b] * dw
                    lr = torch.clip(self.eta[k_b] * self.beta[k_b], 
                                    max=self.eta_init) # NOTE
                    weights_kb = weights_kb + lr * dw

                    # If weight was negative, add it to updated value
                    weights_kb[idc_negative] += self.weights[k_b][idc_negative]
                    
                    # Ensure the weights are within bounds
                    weights_kb = torch.clip(
                        weights_kb,
                        min=self.w_min,
                        max=self.W_MAX,
                        )
                    
                    # Update the weights tensor
                    self.weights[k_b] = weights_kb
                    
                    # Update the learning rate of neuron <k_b>
                    self.eta[k_b] /= self.eta[k_b]**(1/self.eta_decay) + 1

                    # NOTE
                    self.beta[k_b] += self.cst_beta * self.beta[k_b]
            
            # Reset activity traces
            self.trace = torch.zeros_like(self.trace)
            self.t_till_fast = torch.zeros_like(self.t_till_fast)
    
    def next(self):
        """ Called upon transitioning to a new stimulus. """
        super().next()

        # Reset activity traces
        self.trace = torch.zeros_like(self.trace)

        self.t_till_fast = torch.zeros_like(self.t_till_fast)

        # NOTE
        self.beta = torch.ones_like(self.beta)
