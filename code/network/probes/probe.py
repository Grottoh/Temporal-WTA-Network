from typing import Union, Optional
import os

import network.probes.statistic as stat
#from .logger import Logger
#from ..network import Network
from ..component import Component

class Probe:

    """ Probe the (parts of) network for data/diagnostics, in order get an
    impression of its behaviour/evolution. """

    def __init__(
            self,
            component: Component,
            network: 'Network',
            en_show: Union[bool, list[bool]],
            en_save: Union[bool, list[bool]],
            iv_probe: Union[int, float] = float('inf'),
            device: str = 'cpu',
            ) -> None:
        
        # The component to probe, and the network to which it belongs
        self.component = component
        self.network = network

        # Determine whether to show or save (parts of) the probe's data
        self._en_show = [en_show] if type(en_show) == bool else en_show
        self._en_save = [en_save] if type(en_save) == bool else en_save
        
        # The interval at which to probe
        self.iv_probe = iv_probe

        # Device to use when dealing with torch tensors
        self.device = device
        
        # Used to gather experiment results
        self.results = dict()

    def init_directories(self, en_save: bool = True, **kwargs) -> None:
        """ Initialize directories where probe data is to be saved. """
        # If necessary, create directory for the component that is being probed
        self.pd_component = self.network.pd_run + self.component.name + "/"
        if en_save and any(self._en_save):
            if not os.path.isdir(self.pd_component):
                os.mkdir(self.pd_component)

    def add_statistic(self, statistic: stat.Statistic) -> stat.Statistic:
        """ If it doesn't already, have the network track the statistic."""

        # Add an empty statistics dictionary to the network if necessary
        if not hasattr(self.network, 'statistics'):
            self.network.statistics = dict()
        
        # Add the given statistic to the network if is not in there already
        if not statistic.key in self.network.statistics.keys():
            self.network.statistics[statistic.key] = statistic
        else: # If it is there, select one of the two based on their parameters
            self.network.statistics[statistic.key] = type(statistic).select(
                stat_a=self.network.statistics[statistic.key],
                stat_b=statistic
                )
        
        # Return the selected statistic
        return self.network.statistics[statistic.key]

    def en_show(self, i: Optional[int] = None):
        """ Determine whether to show something according to given index. """
        if i == None: # If there is only one thing to show ...
            assert len(self._en_show) == 1 # Assert that there is but one thing
            return self._en_show[0]
        else: # Determine whether to show whatever corresponds to index <i>
            return self._en_show[i]

    def en_save(self, i: Optional[int] = None):
        """ Determine whether to save something according to given index. """
        if i == None: # If there is only one thing to save ...
            assert len(self._en_save) == 1 # Assert that there is but one thing
            return self._en_save[0]
        else: # Determine whether to save whatever corresponds to index <i>
            return self._en_save[i]

    def step(self) -> None:
        """ Called at the end of a timestep. """
        pass # Implemented by child class

    def next(self) -> None:
        """ Called upon transitioning to a new stimulus. """
        pass # Implemented by child class

    def probe(self, ith_stimulus: int, n_stimuli: int) -> None:
        """ If the appropriate interval has passed, or if this is the final
        stimulus of the run: do the probe's thing. """
        if ith_stimulus % self.iv_probe == 0 or ith_stimulus == n_stimuli:
            self._probe(ith_stimulus=ith_stimulus, n_stimuli=n_stimuli)
    
    def _probe(self, ith_stimulus: int, n_stimuli: int) -> None:
        """ Do whatever the (child) probe is meant to do. """
        pass # Implemented by child class