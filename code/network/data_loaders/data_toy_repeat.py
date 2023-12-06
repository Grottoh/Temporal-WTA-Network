from typing import Any, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from contextlib import suppress
import pandas as pd
import os

from network.data_loaders.data_toy_temporal import DataToyTemporal

from .data import Data
from ..utils.util import Constants as C

class DataToyRepeat(Data):
    
    X_1 = torch.cat( (DataToyTemporal.X[0], DataToyTemporal.X[1]), 1)
    X_2 = torch.cat( (DataToyTemporal.X[1], DataToyTemporal.X[0]), 1)
    X_3 = torch.cat( (DataToyTemporal.X[2], DataToyTemporal.X[3]), 1)
    X_4 = torch.cat( (DataToyTemporal.X[3], DataToyTemporal.X[2]), 1)
    X = torch.stack( (X_1, X_2, X_3, X_4), dim=0)
    
    # The pattern names
    T = ['down-up', 'up-down', 'converging-diverging', 'diverging-converging']

    # Path to directory where data files are located
    PD_DATA = C.PD_DATA+"toy-repeat/"
    PF_DATAFRAME = PD_DATA+"toy-repeat.csv"
    PF_X = PD_DATA+"toy-repeat.pt"

    @classmethod
    def to_dataframe(cls) -> None:
        """
        Save static toy data to a pandas dataframe. The data points are saved
        as a single tensor <X> via torch. A separate pandas dataframe is then
        created and saved. This dataframe indicates which index of <X> is
        associated with which pattern.
        """
        # TODO: method is identical that of toy_static; remove repetition

        # Created the data directory if it does not already exist
        if not os.path.isdir(cls.PD_DATA):
            os.mkdir(cls.PD_DATA)
            print(f"\nCreated directory <{cls.PD_DATA}>.\n")
        
        # Create a dataframe for the train labels
        df = pd.DataFrame(
            {'set': [Data.TRAIN for _ in range(cls.X.shape[0])],
             'pattern': cls.T}
            )
        
        # Save the dataframe
        df.to_csv(cls.PF_DATAFRAME, index=False)

        # Stack then save the train and test data
        torch.save(cls.X, cls.PF_X)


    def __init__(
            self,
            label_by: list[str] = ['pattern'],
            **kwargs,
            ) -> None:
        super().__init__(label_by=label_by, **kwargs)
        
        # Determine the shape of the data and the range of its values
        self.shape = (10, 20)
