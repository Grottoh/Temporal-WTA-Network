from typing import Any, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from contextlib import suppress
import pandas as pd
import os

from .data import Data
from ..utils.util import Constants as C, load_pkl

class DataMNIST(Data):

    """ Class for loading MNIST data. """

    # Path to directory where MNIST files are located
    PD_DATA = C.PD_DATA+"mnist/"
    PF_DATA = PD_DATA+"mnist-normal.pkl"
    PF_DATAFRAME = PD_DATA+"mnist_df.csv"
    PF_X = PD_DATA+"mnist_X.pt"

    @classmethod
    def to_dataframe(cls) -> None:
        """
        Save MNIST data to a pandas dataframe. The MNIST images are saved as a
        single tensor <X> via torch. A separate pandas dataframe is then
        created and saved, which indicates which index of <X> should be
        associated with which data subset (train or test) and with which digit.

        Assumes that - before conversion - data is saved to a single pickle
        file as a 4-tuple of numpy arrays.
        """

        # Load MNIST train and test data
        X_train, T_train, X_test, T_test = load_pkl(cls.PF_DATA)
        
        # Create a dataframe for the train labels
        df_train = pd.DataFrame(
            {'index_set': [i for i in range(X_train.shape[0])],
             'set': [Data.TRAIN for _ in range(X_train.shape[0])],
             'digit': T_train}
            )
        
        # Create a dataframe for the test labels
        df_test = pd.DataFrame(
            {'index_set': [i for i in range(X_test.shape[0])],
             'set': [Data.TEST for _ in range(X_test.shape[0])],
             'digit': T_test}
            )
        
        # Create and save a single dataframe for all types of labels
        df = pd.concat([df_train, df_test], ignore_index=True, axis=0)
        df.to_csv(cls.PF_DATAFRAME, index=False)

        # Stack then save the train and test data
        X = torch.tensor( np.vstack([X_train, X_test]), dtype=torch.uint8 )
        torch.save(X, cls.PF_X)

    @staticmethod
    def binarize_pixels(pixels: torch.FloatTensor) -> torch.BoolTensor:
        """ Convert non-binary (grayscale) pixels to binary (black-and-white)
        pixels. """
        pixels[pixels > 0] = 1
        return pixels # TODO: check if should cast to bool or float32
    
    @staticmethod
    def normalize_pixels(pixels: torch.FloatTensor) -> torch.BoolTensor:
        """ Normalize pixels to be in the range [0, 1]. """
        return pixels / torch.max(pixels)
    
    @staticmethod
    def plot_image(x: torch.FloatTensor, t: int) -> None:
        """ Plot a given MNIST image. """
        plt.imshow(x, cmap=plt.get_cmap('gray'))
        plt.title(f"label={t}")
        plt.show()

    def __init__(
            self,
            binary: bool,
            label_by: list[str] = ['digit'],
            **kwargs,
            ) -> None:
        super().__init__(label_by=label_by, **kwargs)
        
        # If True: transform the data to be binary (black-and-white)
        self.binary = binary
        
        # Set the shape of the data and the range of its values
        self.shape = (28, 28)
    
    def load_data(self, **kwargs) -> None:
        super().load_data(**kwargs)
        
        # If enabled, convert the MNIST images to black-and-white images
        if self.binary: self.X = self.binarize_pixels(self.X)
        else: self.X = self.normalize_pixels(self.X)