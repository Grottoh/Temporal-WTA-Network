from typing import Optional
import torch
import numpy as np
from enum import Enum
import math

from .spike_generator import SpikeGenerator
from ..utils.util import get_seed
from ..data_loaders.data import Data

class SGBinned(SpikeGenerator):
    
    LINEAR = "LINEAR"
    FREQUENCY = "FREQUENCY"

    def __init__(
            self,
            dim_time: int,
            n_bins: int = 8,

            sepzero: bool = False,

            omit_radius: float = 0.1,
            
            en_repeat: bool = True,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        # Determine which dimension should become the time dimension
        self.dim_time = dim_time

        # Determine how many bins are used to encode values (positive and
        # negative values are encoded by separated set of <n_bins> bins)
        self.n_bins = n_bins

        # If true, have equal amount of bins for negative and positive values
        self.sepzero = sepzero

        # Omit values that fit in <omit_radius> around zero
        self.omit_radius = omit_radius

        # If True, the same neuron can spike consecutively
        self.en_repeat = en_repeat

    def _init_bins(self):
        
        # Linear: the size of the range covered by each bin is equal
        if self.mode == self.LINEAR:

            # Determine the size of each bin
            # NOTE: assumes data is normalized to range [-1, 1] !!!
            bin_size = 2.0 / self.n_bins
                    
            # Bins are separated at zero and both sides must have an equal
            # number of bins, therefore <n_bins> must be even
            if self.n_bins%2 != 0:
                raise ValueError(f"<n_bins={self.n_bins}> must be even when" +
                                 f" <sepzero={self.sepzero}> is True.")

            # <omit_radius> must be smaller than bin size (perhaps not crucial)
            if self.omit_radius >= bin_size:
                raise ValueError(f"<bin_min={self.omit_radius}> must be"
                                 f" smaller than <bin_size={bin_size}.")

            # Create an array for the bins
            self.bins = np.zeros(
                shape=(self.n_bins, 2),
                dtype=np.float32,
                )

            # First bin starts at -1
            self.bins[0] = np.array( 
                [-1, -1+bin_size], 
                dtype=np.float32,
                )
            
            # The remaining bins are added according to <bin_size>
            for i in range(1, self.n_bins):
                start = self.bins[i-1][1] # <start> is <stop> of previous bin
                stop  = start + bin_size
                self.bins[i] = np.array( 
                    [start, stop], 
                    dtype=np.float32,
                    )
                
        # Frequency: the size of the range covered by each bin is determined by
        # the frequency of values within that range (more frequent values
        # warrant more precision and thus smaller bins)
        elif self.mode == self.FREQUENCY:

            # Load the bins stored with the data
            self.bins = self.layer.data.load_bins(
                sepzero=self.sepzero,
                omit_radius=self.omit_radius,
                n_bins=self.n_bins,
            )
        
        else:
            raise NotImplementedError

    
    def set_shape(self):
        """ Determine the shape of the to be generated spikes. """

        # Convert shape to list so it can be altered
        shape = list(self.layer.data.shape)

        # Remove time dimension from shape and determine its size
        self.n_timesteps = shape.pop(self.dim_time)

        # Set time dimension as the first dimension
        self.shape = ((self.n_timesteps,) + 
                      tuple([s*self.n_bins for s in shape]))
                       # NOTE - was 2*self.n_bins; LINEAR won't work  now
                       # anymore (also for other reasons)

        # Initialize the bins
        self._init_bins()
    
    def generate_spikes(self, X_i: torch.FloatTensor) -> torch.FloatTensor:
        """ Convert the given data sample to spikes. """

        def _find_spike(bin: np.ndarray, value: float) -> bool:
            """ Return True if the value is in the bin range. """
            return (value >= bin[0] and value < bin[1])
        
        def find_spike(value: np.ndarray, bins) -> np.ndarray:
            """ Return bool array indicating which bin is active. """
            return np.apply_along_axis(_find_spike, 1, bins, value)

        # NOTE: In order to get this to for n-dimensional input I'll have to
        # change stuff. Currently it works only for 1D input + a time dimension

        S_i = np.empty(shape=self.shape, dtype=bool)
        
        # Linear: each bin is of the same size
        if self.mode == self.LINEAR:
            intensity = X_i.swapaxes(self.dim_time, 0).cpu().numpy()
            
            # Determine which bins are active for each value at each time
            for t, values in enumerate(intensity):
                values = np.expand_dims(values, 1)
                spikes_t = np.apply_along_axis(
                    find_spike, 1, values, self.bins)

                # Omit values around zero that fall within <omit_radius>
                # NOTE: this implementation makes it so that coefficient <0>
                # can not be used (given that there the values are not centered
                # around zero)
                if self.omit_radius != None:
                    omit_mask = ((values < -self.omit_radius) | 
                                 (values > self.omit_radius))
                    spikes_t *= omit_mask[:, :, None]
                
                spikes_t = spikes_t.reshape(self.shape[1])
                
                S_i[t] = spikes_t
        
        # Frequency: size of bin increases as values within are less frequent
        elif self.mode == self.FREQUENCY:
            intensity = X_i.swapaxes(self.dim_time, 0).cpu().numpy()
            
            # Determine which bins are active for each value at each time
            for t, values in enumerate(intensity):
                for i, coefficient in enumerate(values):
                    # TODO: APPLY ALONG TIME AXIS IF POSSIBLE?
                    
                    # Determine which bin is active for the coefficient value
                    # If the value falls withing <omit_radius> don't spike
                    # NOTE: this implementation makes it so that coefficient
                    # <0> can not be used (given that there the values are not
                    # centered around zero)
                    if (self.omit_radius != None and 
                        (coefficient >= -self.omit_radius and 
                         coefficient <= self.omit_radius)):
                            spike_t_i = np.zeros(self.n_bins, dtype=bool)
                    
                    # Trigger a spike at the appropriate bin
                    else:
                        bins_i = self.bins[i]#[] # NOTE
                        spike_t_i = np.apply_along_axis(
                            _find_spike, 1, bins_i, coefficient)
                    
                    S_i[t, i*self.n_bins:i*self.n_bins+self.n_bins] = spike_t_i
        
        # Unknown: raise an error
        else:
            raise ValueError(f"SGStatic mode <{self.mode}> is not known.")

        if not self.en_repeat:

            # Determine where values change with respect to previous values
            change = np.diff(S_i, axis=0) != 0

            # Only allow spikes where values have changed
            S_i = np.concatenate( (S_i[0, None], S_i[1:]*change), axis=0 )
        xxx = np.array(S_i, dtype=np.uint8) # NOTE
        S_i = torch.from_numpy(S_i).to(dtype=torch.float32, device=self.device)
        
        return S_i