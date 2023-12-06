from typing import Optional
import torch
import numpy as np

import network.probes.statistic as stat
from network.weights.weights import Weights
from .logger import Logger
from ..layers.layer import Layer

class LoggerWeights(Logger):
    
    def __init__(
            self,
            weights: Weights,
            thresholds: list[float] = [0.0, 1e-3, 1e-2, 1e-1, 0.5, 0.9, 0.99],
            **kwargs
            ) -> None:
        super().__init__(component=weights, **kwargs)

        self.thresholds = thresholds
        self.n_weights_total = torch.numel(weights.weights)
    
    @property
    def weights(self):
        """ Alternative reference to target (weights) component. """
        return self.component

    def _probe(self, ith_stimulus: int, n_stimuli: int) -> None:

        # Do nothing if none of the loggers are active
        if not self.active:
            return
        
        msg = ""

        msg += (f"[{Logger.now()}] - {ith_stimulus:5d} / {n_stimuli}" +
                f" - {self.weights.name} - {self.weights.host.name}\n")
        
        for threshold in self.thresholds:
            n_weights = int( torch.sum( self.weights.weights <= threshold ) )
            msg += f"Threshold={threshold:.4f}"
            msg += " ---"
            msg += f" n_weights: {n_weights:,} / {self.n_weights_total:,}"
            msg += f" ({n_weights/self.n_weights_total * 100:.2f}%)"
            msg += "\n"
            self.results[threshold] = n_weights # Gather experiment results
            
            if threshold == 0.0:
                n_nonzero = self.n_weights_total - n_weights
                msg += " > "
                msg += f"Non-zero: {n_nonzero:,} / {self.n_weights_total:,}"
                msg += f" ({n_nonzero/self.n_weights_total * 100:.2f}%)"
                msg += "\n"
                self.results['nonzero'] = n_nonzero # Gather experiment results
        self.results['total'] = self.n_weights_total# Gather experiment results
        
        msg += "\n"

        self.produce(msg=msg)
    

    
