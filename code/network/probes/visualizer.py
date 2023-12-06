from __future__ import annotations
from typing import Optional, Union
from collections import namedtuple
import datetime
import os

from ..utils.util import Constants as C
from .probe import Probe

class Visualizer(Probe):

    """ Show or save visualizations of data/diagnostics regarding (parts of)
    the network. """
    
    def __init__(
            self,
            **kwargs,
            ) -> None :
        super().__init__(**kwargs)

    def init_directories(self) -> None:
        """ Initialize directory where component's visuals are to be saved. """
        
        # Initialize directory for the target component
        super().init_directories(en_save=self.network.en_save_visuals)
        
        # If necessary, create a directory for the component visuals
        self.pd_visuals = self.pd_component + C.ND_VISUALS
        if self.network.en_save_visuals and any(self._en_save):
            if not os.path.isdir(self.pd_visuals):
                os.mkdir(self.pd_visuals)
    
    def en_show(self, i: Optional[int] = None):
        """ Determine whether to show something according to given index. """
        return self.network.en_show_visuals and super().en_show(i=i)
    
    def en_save(self, i: Optional[int] = None):
        """ Determine whether to save something according to given index. """
        return self.network.en_save_visuals and super().en_save(i=i)