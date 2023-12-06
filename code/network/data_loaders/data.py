from typing import Any, Union, Optional
import torch
import pandas as pd
from enum import Enum
from contextlib import suppress
import numpy as np


from ..component import Trait

class Data(Trait):

    # Determine what type of data to use, can be expanded upon by child classes
    TRAIN = "train"
    TEST = "test"

    def __init__(
            self,
            label_by: list[str],
            whitelists: list[ dict[str, list[Any]] ] = [dict()],
            blacklists: list[ dict[str, list[Any]] ] = [dict()],
            shuffle: bool = False,
            _T: Optional[np.ndarray] = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        
        # Create unique labels based on the selected columns
        self.label_by = label_by

        # Select only rows that satisfy the conditions of the whitelists
        self.whitelists = whitelists
        
        # Remove any row that satisfies the conditions of the blacklists
        self.blacklists = blacklists

        # If enabled, shuffle the data
        self.shuffle = shuffle

        # Mapping from tuple-of-column-values to label
        self._T = _T
    
    @property
    def saveables(self) -> dict[str, Any]:
        """<__dict__> exluding attributes that can be saved, but shouldn't. """
        forgettables = [] # Specifies the to be forgotten attributes
        with suppress(AttributeError): forgettables.append(self.df)
        with suppress(AttributeError): forgettables.append(self.X)
        with suppress(AttributeError): forgettables.append(self.T)
        return self._saveables(forgettables=forgettables)
    
    def load_data(
            self,
            sets: list[str],
            n_stimuli: Union[int, float] = float('inf'),
            ) -> None:
        
        # Load single dataframe associating indices with labels
        self.df = pd.read_csv(self.PF_DATAFRAME)
        
        # Load all data
        self.X = torch.load(self.PF_X).to(
            dtype=torch.float32, device=self.device)
        
        # Determines which set(s) to use (e.g. train/test/validation/... set)
        self.df = self.df.loc[ self.df['set'].isin(sets) ]

        # Select only rows that satisfy the whitelist conditions
        for whitelist in self.whitelists:
            for col_header, values in whitelist.items():
                if col_header == 'set':
                    raise ValueError(
                        f"Whitelist should not be used to filter <set>.")
                elif len(values) == 0:
                    raise ValueError(
                        f"Whitelist filter for column <{col_header}>" +
                         " must not be empty")
                self.df = self.df.loc[ self.df[col_header].isin(values) ]

        # Remove any row that satisfies the blacklist conditions
        for blacklist in self.blacklists:
            for col_header, values in blacklist.items():
                self.df = self.df.loc[ ~self.df[col_header].isin(values) ]

        # If enabled, shuffle the dataframe
        if self.shuffle:
            self.df = self.df.sample(
                n=len(self.df.index), 
                replace=False,
                random_state=self.seed,
                )
            
        # Select at most <n_stimuli> samples
        self.n_samples = min(len(self.df.index), n_stimuli)
        self.df = self.df.head(n=self.n_samples)
        
        # Select only columns that are relevant for the label
        if isinstance(self._T, type(None)): # If it isn't set already

            # Create a new column of unique labels based on selected columns
            self.df['label'] = self.df.groupby(
                self.df[self.label_by].apply(frozenset, axis=1),
                ).ngroup()
            
            # Determine the number of unique class labels
            self.n_classes = self.df['label'].nunique()

        #     xxx = self.df.set_index(
        #         self.label_by)['label']
        #    # yyy = xxx.astype({'index':'string'})
        #     xxx['index'] = xxx["index"].map(str)
        #     #xxx['index'] = 

            # TODO: clean up
            # Mapping from tuple-of-column-values to label
            temp: [ Union[Any, tuple[Any]], int ] = self.df.set_index(
                self.label_by)['label'].to_dict()
            
            # Ensure keys are strings (easier when loading from JSON)
            self._T: dict[str, int] = dict()
            for key, value in temp.items():
                self._T[str(key)] = value
        else:
            self.n_classes = len(self._T)

        # Select the appropriate data in the appropriate order
        self.X = self.X[self.df.index.values]

        # Assign the appropriate label to each sample
        self.T = np.empty(shape=self.n_samples, dtype=np.uint16)
        for idx, (_, row) in enumerate(self.df.iterrows()):

            # Retrieve the column values that determine the label
            _label_by = tuple([ row[header] for header in self.label_by ])
            _label_by = (str(_label_by[0]) if 
                         len(_label_by) == 1 else 
                         str(_label_by))

            # If the key/label is yet unknown, add it
            if not _label_by in self._T:
                print(f"<_label_by={_label_by}> is not known, assigning it" + 
                      f" new label <{len(self._T)}>.")
                self._T[_label_by] = len(self._T)
                self.n_classes += 1
            
            # Determine the label of sample <idx>
            self.T[idx] = self._T[_label_by]
        
        self.T_ = {v: k for k, v in self._T.items()}
        # Reset the indices such that they range [0, n_samples)
        #df = df.reset_index(drop=True) # TODO: decide whether this is useful