from typing import Optional, Callable
import torch

from ..component import Trait

class Weights(Trait):

    def __init__(
            self,
            shape: tuple[int, ...],
            w_min: Optional[float],
            w_max: Optional[float],
            eta_init: Optional[float] = 1.0,
            eta_decay: Optional[float] = 1.0,
            eta_star: int = 1,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
        self.shape = shape # The shape of the weights

        self.w_min = w_min # The minimum value of weights
        self.w_max = w_max # The maximum value of weights

        self.eta_init = eta_init   # Initial learning rate
        self.eta_decay = eta_decay # Lower value -> slower learning rate decay
        self.eta_star = eta_star # Number of times to repeat the learning step

    def generate_weights(
            self,
            weight_generator: Callable[[tuple[int, ...],
                                        float,
                                        float,
                                        Optional[str], 
                                        Optional[torch.Generator]], 
                                        torch.FloatTensor],
            ):
        """ Generate weights according to host axon and weight generator. """
        self.weights = weight_generator(
            shape=self.shape,
            w_min=self.w_min,
            w_max=self.w_max,
            device=self.device,
            rng=self.rng,
            )
        
        # Initialize the learning rate of each axon group
        self.eta = torch.full(
            size=self.host.shape_b,
            fill_value=self.eta_init,
            dtype=torch.float32,
            device=self.device,
            )

    
    def learn(self):
        """ Base weights do not learn. """
        super().learn()
        pass

class WeightGenerator:

    """ A variety of functions for generating weights. """

    def uniform(
            shape: tuple[int, ...],
            init_min: float,
            init_max: float,
            w_min: float,
            w_max: float,
            relative: bool,
            device: str = 'cpu',
            rng: Optional[torch.Generator] = None,
            ) -> torch.FloatTensor:
        """ Generate weights uniformly within the specified range. """
        
        # Assert that the upper limit is higher than the lower limit
        assert w_min < w_max

        if relative: # Convert relative initial weight limits to be absolute
            init_min = w_min + init_min * (w_max - w_min)
            init_max = w_min + init_max * (w_max - w_min)
        
        # Assert that the initial weight limits do not exceed the true limits
        assert w_min <= init_min <= w_max and w_min <= init_max <= w_max

        # Generate weights uniformly in the range of [init_min, init_max)
        weights = torch.empty(
            size=shape, 
            dtype=torch.float32,
            device=device,
            ).uniform_(init_min, init_max, generator=rng)
        
        return weights

    def gaussian(
            shape: tuple[int, ...],
            init_mean: float,
            init_stdv: float,
            w_min: float,
            w_max: float,
            relative: bool,
            device: str = 'cpu',
            rng: Optional[torch.Generator] = None,
            ) -> torch.FloatTensor:
        """ Generate weights according to a Gaussian. """
        
        # Assert that the upper limit is higher than the lower limit
        assert w_min < w_max

        if relative: # Convert relative initial weight limits to be absolute
            init_mean = w_min + init_mean * (w_max - w_min)
            init_stdv = init_stdv * (w_max - w_min)
        
        # Assert that the initial mean is not to close to the weight limits
        assert (w_min + init_stdv) <= init_mean <= (w_max - init_stdv)

        # Generate weights from a Gaussian
        weights = torch.normal(
            mean=init_mean,
            std=init_stdv,
            size=shape,
            device=device,
            generator=rng,
            )
        
        # Ensure that weights to not exceed the given limit
        weights = torch.clip(weights, min=w_min, max=w_max)

        return weights       