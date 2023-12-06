from typing import Optional, Union
import torch

from ..data_loaders.data import Data
from ..component import Trait

class SpikeGenerator(Trait):
    """ A base SpikeGenerator implementation, assumes data samples are already
    formulated in terms of spikes. """

    def __init__(
            self,
            mode: str,
            shape: Union[ tuple[int, ...], list[int] ] = tuple(),
            n_timesteps: Optional[int] = None,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        # TODO: even though when creating a new network, <shape> and
        # <n_timesteps> is set using <set_shape>, right now I include <shape>
        # and <n_timesteps> in the parameters because this is necessary to
        # initialize them properly after loading. I want to look into a more
        # elegant solution however. This is a thing in other classes as well

        # TODO: do I want shape to be listable?
        self.shape = tuple(shape)
        self.n_timesteps = n_timesteps

        # Set the operating mode of the spike generator
        self.mode = mode
    
    @property
    def layer(self):
        """ Alternative reference to host component, which is a layer. """
        return self.host
    
    def set_shape(self):
        """ Determine the shape of the to be generated spikes. """
        self.shape = self.layer.data.shape

    def generate_spikes(self, X_i: torch.FloatTensor) -> torch.FloatTensor:
        """ Return <X_i>, which is already formulated in terms of spikes """
        return X_i