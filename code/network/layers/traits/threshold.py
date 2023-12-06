from typing import Union, Callable, Optional
import torch
import numpy as np

from ...component import Trait

class Threshold(Trait):
    """ Membrane potential threshold that remains constant. """

    def __init__(
            self,
            mps_max: Union[float, torch.FloatTensor],
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
        self.mps_max = mps_max
    
    @property
    def layer(self):
        """ Alternative reference to host component, which is a WTA layer. """
        return self.host
    
    def step(self):
        super().step()
        pass # Do not change the membrane potential threshold

class ThresholdAdaptive(Threshold):

    def __init__(
            self,
            hz_timestep: int, 
            hz_layer: int, 
            cst_decay: float = 1.0,
            cst_growth: float = 1.0,
            **kwargs
            ) -> None:
        super().__init__(mps_max=None, **kwargs)

        # Aim for a spike every <iv_spike> timesteps
        self.hz_timestep = hz_timestep
        self.hz_layer = hz_layer
        self.iv_spike = self.hz_timestep / self.hz_layer

        # Initialize trace at max value, correlates with time since last spike
        self.trace = 1.0
        
        # Constant that determines the decay of membrane potentials
        self.cst_decay = cst_decay # <1.0> means no decay
        if not ( 0.0 < self.cst_decay <= 1.0 ): # Be in range (0, 1]
            raise ValueError(f"Threshold decay constant <{self.cst_decay}>" +
                             f" is outside of range (0, 1].")
        
        # Constant that determines the growth of membrane potentials
        self.cst_growth = cst_growth # Growth is at most <mp_max*cst_growth>
        if not self.cst_growth > 0.0: # Be in range (0, inf)
            raise ValueError(f"Threshold growth constant <{self.cst_growth}>" +
                             f" is outside of range (0, inf).")

    def generate_weights(
            self,
            weight_generator: Callable[[tuple[int, ...],
                                        Optional[str], 
                                        Optional[torch.Generator]], 
                                        torch.FloatTensor],
            ):
        """ Generate weights according to host axon and weight generator. """
        self.mps_max = weight_generator(
            shape=self.host.shape,
            device=self.device,
            rng=self.rng,
            )
    
    def learn(self):
        """ Increase threshold potentials if necessary. """
        super().learn()
        if np.random.randint(0, 5000) < 1:
            print(self.mps_max)
        if self.layer.spiked: # If a neuron in the layer spiked
            for k in self.layer.ks: # Iterate over spiking neurons <ks>

                # Determine by how much neuron <k>'s <mp_max> should increase
                growth = self.trace * self.mps_max[k] * self.cst_growth

                # If there is any trace left, increase threshold of neuron <k>
                self.mps_max[k] += growth
                
            # Set the trace at max value
            self.trace = 1.0
            
        else: # Otherwise have the trace decay
            # NOTE: does this require re-evaluation (especially the constant,
            # which I originally set to 9/15)
            decay = np.exp(self.trace - 1) / (9/15 * self.iv_spike)
            self.trace = max( 0, self.trace - decay)

            # Lower mp thresholds according to <trace> and <cst_decay>
            if self.trace < 0.01: # TODO: is this how I want to define it?
                self.mps_max *= self.cst_decay
        pass
    
    def next(self):
        """ On a new stimulus, reset the trace to its maximum value. """
        super().next()
        self.trace = 1.0
            
class ThresholdDependent(Threshold):

    def __init__(
            self,
            hz_a: int,
            hz_b: int,
            **kwargs
            ) -> None:
        super().__init__(mps_max=None, **kwargs)
        
        self.hz_a = hz_a
        self.hz_b = hz_b
        
        self.mps_max = torch.ones(
            size=self.layer.shape,
            dtype=torch.float32,
            device=self.device
            )

    def learn_two(self):
        super().learn_two()
        if self.layer.spiked:
            for k in self.layer.ks:
                self.mps_max[k] = 0
                for axon in self.layer.axons_a:
                    weights = axon.weights[k]
                    eta = axon._weights.eta[k]

                    xxx =torch.sum(weights[weights>0])

                    # NOTE: Only consider positive weights
                    self.mps_max[k] += torch.sum(weights[weights>0])
                    #self.mps_max[k] /= (1+10*eta)
                    #self.mps_max[k] /= len(self.layer.axons_a)
                    self.mps_max[k] *= self.hz_a / self.hz_b
                    
                    # self.mps_max[k] = torch.clip(self.mps_max[k], 
                    #                              min=self.hz_a / self.hz_b)
                    
                    self.mps_max[k] = torch.clip(self.mps_max[k], 
                                                 min=1)
