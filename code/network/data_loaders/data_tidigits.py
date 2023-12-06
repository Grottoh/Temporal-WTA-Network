import json
from typing import Any, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from contextlib import suppress
import pandas as pd
import os
import scipy.io
import librosa
import soundfile as sf
import math
import re

from .data import Data
from ..utils.util import Constants as C

# NOTE: I currently have access to only a small subset of the TIDIGITS data,
# with only about 2500 train, and 2500 test samples
class DataTidigits(Data):

    """ Class for loading TIDIGITS data. """

    # The duration of the longest audio file
    #MAX_DURATION = 51200

    # TIDIGITS sampling rate
    SAMPLING_RATE = 20_000

    # The amount of samples we are shifting after each fft (wrt MFCCs)
    HOP_LENGTH = 512
    
    # The amount of MFCCs to use
    N_MFCCS = 13

    # Path to directory where TIDIGITS files are located
    PD_DATA = C.PD_DATA+'tidigits/'
    PD_DATA_OG = PD_DATA+'tidigits_flac/data/'
    PF_DATAFRAME = PD_DATA+"dataframe.csv"

    # Path to file where raw audio is stored (as array of 1D arrays)
    PF_X_RAW = PD_DATA+"X_raw.npy"

    # Path to file where MFCCs are stored as torch tensors (including deltas)
    PF_X = PF_X_MFCC = PD_DATA+"X_mfcc.pt"

    # Path to file where a large variety of MFCC bins are stored
    PF_BINS = PD_DATA+"bins.json"

    # Template for selecting various types of MFCC bins
    KEY_BINS = "sepzero-{}_omit-{}_n-{}_form-{}_coef-{}"

    KEY_INDEX = 'index'
    KEY_SPEAKER = 'speaker-id'
    KEY_SET = 'set'
    KEY_AGE = 'age'
    KEY_DEVELOPMENT = 'development'
    KEY_SEX = 'sex'
    KEY_DIALECT = 'dialect'
    KEY_N_SAMPLES = 'n-samples'
    KEY_N_TIMEBINS = 'n-timebins'
    KEY_UTTERANCE = 'utterance'
    KEY_N_DIGITS = 'n-digits'

    @classmethod
    def to_dataframe(cls) -> None:
        """
        Save TIDIGITS data to a pandas dataframe. The TIDIGITS audio is saved
        via torch as ...
        """
        
        def get_speakerinfo():

            # Path to text file with speaker info (both adults and children)
            PF_SPEAKERINFO = cls.PD_DATA_OG+'adults/doc/spkrinfo.txt'

            speaker2info: dict[ str, dict[str, Union[str, int]] ] = dict()
            with open(PF_SPEAKERINFO, 'r') as f:
                for line in f:

                    # Do not process comments
                    if line.startswith(';'):
                        continue

                    #re.split("\s+", line)
                    split = line.split()

                    # Retrieve the speaker id
                    id_speaker = split[0].lower()

                    # Retrieve the speaker info
                    info = {
                        cls.KEY_SEX: ('female' if 
                                      split[1] in ['G', 'W'] else 
                                      'male'),
                        cls.KEY_AGE: int(split[2]),
                        cls.KEY_DEVELOPMENT: ('child' if 
                                              int(split[2]) < 17 else 
                                              'adult'),
                        cls.KEY_DIALECT: int(split[3]),
                        cls.KEY_SET: 'train' if split[4] == 'TRN' else 'test',
                        }

                    speaker2info[id_speaker] = info
                
            return speaker2info
        
        def create_bins(X_list: list[np.ndarray]):

            def add_bin(
                X_list: list[np.ndarray],
                name2bins: dict[ str, list[ list[float, float] ] ],
                sepzero: bool,
                omit_radius: float,
                n_bins: int,
                form: str,
                idx_coef: int,
                ) -> None:
                    
                    """ Add a bin list to bin dictionary <name2bins>. """
                    
                    def decide_bin(
                        coef_i: np.ndarray, 
                        n_bins: int,
                        ) -> list[ list[float, float] ]:

                        """ Decide the bin list for a single coefficient. """

                        # Total number of values of coefficient <i> in the data
                        n_values = coef_i.size

                        # Number of values per bin for coefficient <i>
                        n_values_per_bin = n_values // n_bins

                        # Determine the bins of coefficient <i>
                        bins = []
                        idx_start = 0
                        idx_stop = n_values_per_bin - 1
                        for idx_bin in range(n_bins):

                            # Account for rounding
                            if idx_bin == n_bins-1:
                                idx_stop = coef_i.size - 1

                            # Determine at which value the bin starts and stops
                            bin_start = float(coef_i[idx_start])
                            bin_stop  = float(coef_i[idx_stop])
                            
                            # Add the bin to the coefficient bins
                            bins.append([bin_start, bin_stop])

                            idx_start = idx_stop
                            idx_stop = idx_start + n_values_per_bin - 1

                        return bins
                    
                    # If bins are separated at zero, both sides must have an
                    # equal number of bins, therefore <n_bins> must be even
                    if sepzero and n_bins%2 != 0:
                        return # Do not create bins in this case

                    # Select and sort all values of the appropriate coefficient
                    if form.lower() == 'mfcc': # Select MFCCs
                        coef_i = [mfcc[idx_coef] for mfcc in X_list]
                        coef_i = np.sort( np.concatenate(coef_i) )
                        is_mfcc = True
                    elif form.lower() == 'delta': # Select MFCC deltas
                        coef_i = [mfcc[cls.N_MFCCS+idx_coef] for mfcc in X_list]
                        coef_i = np.sort( np.concatenate(coef_i) )
                        is_mfcc = False
                    else: # Other types of coefficients are not recognized
                        raise NotImplementedError
                    
                    # If enabled, omit certain values
                    if not omit_radius == None:

                        # MFCC zero is strictly negative: omit most
                        # extreme negative values
                        if idx_coef == 0:
                            omit_till = np.min(coef_i) + omit_radius
                            coef_i = coef_i[np.where(coef_i>omit_till)]
                        else: # Otherwise omit around zero
                            coef_i = coef_i[np.where((coef_i<-omit_radius) | 
                                                     (coef_i> omit_radius))]
                    
                    # Create bins for negative and positive values separately
                    # NOTE: MFCC <0> does not have positive values
                    if ( sepzero and not (is_mfcc and idx_coef == 0) ):
                        bins_negative = decide_bin(
                            coef_i=coef_i[np.where(coef_i<0)],
                            n_bins=n_bins//2,
                            )
                        bins_positive = decide_bin(
                            coef_i=coef_i[np.where(coef_i>= 0)],
                            n_bins=n_bins//2,
                            )
                        bins = bins_negative + bins_positive
                    else: # No distinction between negative and positive bins
                        bins = decide_bin(coef_i=coef_i, n_bins=n_bins)
                    
                    # First and last bin stretch to (respectively -/+) infinity
                    bins[0][0]   = float('-inf')
                    bins[-1][-1] = float(' inf')
                    
                    # Determine the name (key) of the coefficient bins
                    name = cls.KEY_BINS.format(
                        sepzero, omit_radius, n_bins, form, idx_coef).lower()
                    print(name)
                    
                    # Add the coefficient bins to the dictionary
                    name2bins[name] = bins

            # For each coefficient and its delta create a variety of bins
            name2bins: dict[ str, list[ list[float, float] ] ] = dict()
            for sepzero in [False, True]:
                for omit_radius in [None, 0.00, 0.01, 0.10, 1.00, 5.00]:
                    for n_bins in range(2, 30):
                        for form in ['mfcc', 'delta']:
                            for idx_coef in range(0, cls.N_MFCCS):
                                add_bin(
                                    X_list=X_list,
                                    name2bins=name2bins, 
                                    sepzero=sepzero,
                                    omit_radius=omit_radius,
                                    n_bins=n_bins,
                                    form=form,
                                    idx_coef=idx_coef,
                                    )
            
            return name2bins
        
        def pad_to_torch(
                X_list: list[np.ndarray],
                max_duration: int,
                ) -> torch.FloatTensor:
            """ Pad the given list of arrays to have the same duration
            (<max_duration), and return it as a torch tensor. """
            
            X_tensor = torch.empty(
                size=(len(X_list),) + X_list[0].shape[:-1] + (max_duration,), 
                dtype=torch.float32
                )

            # Pad each sample in X
            for i, x in enumerate(X_list):

                if max_duration < x.shape[-1]:
                    raise ValueError(
                        f"<max_duration={max_duration}> must be greater or " +
                        f" to the duration of x_{i} ({x.shape[-1]})."
                        )

                pad_width = max_duration - x.shape[-1]

                x = torch.tensor(x, dtype=torch.float32)
                x = torch.nn.functional.pad(x, pad=(0, pad_width,0,0), value=0)

                X_tensor[i] = x
            
            return X_tensor
        
        speaker2info = get_speakerinfo()
        df = pd.DataFrame({})
        X_raw = []
        X = []
        index = 0
        n_samples_max = n_timebins_max = 0
        for nd_set in ['train/', 'test/']:

            for nd_development in ['children/', 'adults/']:
                
                for nd_sex in (['boy/', 'girl/'] if 
                               nd_development == 'children/' 
                               else ['man/', 'woman/']):
                    
                    pd_speakers = cls.PD_DATA_OG+nd_development+nd_set+nd_sex
                    for id_speaker in os.listdir(pd_speakers):
                        pd_speaker = pd_speakers + id_speaker + '/'
                        for flac in sorted(os.listdir(pd_speaker)):

                            # Load raw speech corresponding current utterance
                            raw, _ = librosa.load(pd_speaker+flac, 
                                                        sr=cls.SAMPLING_RATE)
                            n_samples = raw.size
                            n_samples_max = max(n_samples, n_samples_max)
                            X_raw.append(raw)

                            mfccs = librosa.feature.mfcc(
                                y=raw, 
                                sr=cls.SAMPLING_RATE, 
                                hop_length=cls.HOP_LENGTH,
                                n_mfcc=cls.N_MFCCS,
                                )
                            deltas = librosa.feature.delta(mfccs)
                            n_timebins = math.ceil(mfccs.shape[1])
                            n_timebins_max = max(n_timebins, n_timebins_max)
                            x = np.concatenate( (mfccs, deltas), axis=0 )
                            X.append(x)

                            utterance = flac.removesuffix('.flac')[:-1]
                            n_digits = len(utterance)
                            
                            _row = (speaker2info[id_speaker] | 
                                   {cls.KEY_INDEX: index,
                                    cls.KEY_SPEAKER: id_speaker,
                                    cls.KEY_N_SAMPLES: n_samples,
                                    cls.KEY_N_TIMEBINS: n_timebins,
                                    cls.KEY_N_DIGITS: n_digits,
                                    cls.KEY_N_DIGITS: n_digits,
                                    cls.KEY_UTTERANCE: utterance}
                                   )
                            row = pd.DataFrame(_row, index=[0])
                            
                            df = pd.concat([df, row], ignore_index=True)

                            index += 1

                    #         break
                    #     break
                    # break
        
        # Save the dataframe
        df.to_csv(cls.PF_DATAFRAME, index=False)
        
        # Save the raw audio
        np.save(cls.PF_X_RAW, np.array(X_raw, dtype=object))

        # Create and save MFCC&delta bins to a single .json file
        name2bins = create_bins(X_list=X)
        with open(cls.PF_BINS, "w") as f:
            json.dump(name2bins, fp=f, indent=4)

        # Save the MFCCs and their deltas
        X = pad_to_torch(
            X_list=X,
            max_duration=n_timebins_max,
            )
        torch.save(X, cls.PF_X)        
        
    @classmethod
    def plot_mfccs(cls, x: np.ndarray, t: int, duration: int):
        #x = np.flip(x, axis=0)
        librosa.display.specshow(
            x[:, :duration], 
            sr=cls.SAMPLING_RATE, 
            x_axis='time',
            cmap='magma', 
            hop_length=cls.HOP_LENGTH,
            )
        plt.colorbar(label='')
        plt.title(f"label={t}")
        plt.xlabel('Time', fontdict=dict(size=12))
        plt.ylabel('Coefficient', fontdict=dict(size=12))
        #plt.axvline(x=duration/cls.SAMPLING_RATE)
        #plt.savefig(f"norm-v2_{t[0][0]}_{duration}_{np.max(x):.0f}.png")
        plt.savefig(f"norm-v2_{t}_{duration}_{np.max(x):.0f}.png")
        #plt.show()
        plt.close()
        
    
    def __init__(
            self,
            n_timebins_max: int = 60,#TODO:set automatically everywhere somehow
            form='mfcc&delta',
            coefficients: np.ndarray = np.array( range(1, N_MFCCS) ),
            normalization: Optional[str] = None,
            label_by: list[str] = ['utterance'],
            **kwargs,
            ) -> None:
        super().__init__(label_by=label_by, **kwargs)

        # Determine the maximum duration in terms of MFCC timebins
        self.n_timebins_max = n_timebins_max

        # Determine whether to load MFCCs, their deltas, or both
        self.form = form

        # Determine by index which coefficients to load
        self.coefficients = coefficients

        # Determine whether and how to normalize the data
        self.normalization = normalization

        # Determine how many coefficients are selected
        n_coefficients = coefficients.size
        if self.form.lower() == 'mfcc&delta':
            n_coefficients *= 2

        # Determine the shape of the data
        self.shape = (n_coefficients, n_timebins_max)
    
    def load_bins(
            self, 
            sepzero: bool = False,
            omit_radius: float = 0.0,
            n_bins: int = 10,
            ) -> np.ndarray:
        
        def add_bins(
                key: str, 
                bins_all: dict[ str, list[ list[float, float] ] ], 
                bins: list[ list[ list[float, float] ] ],
                ) -> None:
            """ Select bins from <bins_all> and add them to <bins>. """
            for idx_coef in self.coefficients:
                key_i = key.format(idx_coef)
                bins.append( bins_all[key_i] )

        # Load the bins that were saved via JSON
        with open(self.PF_BINS, 'r') as f:
            bins_all = json.load(f)
        
        # Select the appropriate bins
        bins = []
        key = self.KEY_BINS.format(
            sepzero, omit_radius, n_bins, '{}','{}').lower()
        
        # Select only MFCC bins
        if self.form.lower() == 'mfcc':
            key = key.format('mfcc', '{}')
            add_bins(key=key, bins_all=bins_all, bins=bins)
        
        # Select only MFCC delta bins
        elif self.form.lower() == 'delta':
            key = key.format('delta', '{}')
            add_bins(key=key, bins_all=bins_all, bins=bins)
        
        # Select bins for both the MFCCs and their deltas
        elif self.form.lower() == 'mfcc&delta': 
            key_mfcc = key.format('mfcc', '{}')
            add_bins(key=key_mfcc, bins_all=bins_all, bins=bins)
            key_delta = key.format('delta', '{}')
            add_bins(key=key_delta, bins_all=bins_all, bins=bins)
        
        else:
            raise ValueError(f"TIDIGITS data form <{self.form}> is not known.")
        
        # Return the bins as a numpy array
        bins = np.array(bins, dtype=np.float32)
        print(f"Shape of bins: {bins.shape}")

        return bins
    
    def load_data(self, **kwargs) -> None:

        # Load/select the appropriate data
        super().load_data(**kwargs)

        # <X> should contain <N_MFCCS> coefficients (once for MFCCs and once
        # for deltas, hence division by of shape by 2)
        if self.X.shape[1]//2 != self.N_MFCCS:
            raise ValueError(
                f"Number of MFCCs in <X> ({self.X.shape[1]//2}) does not"
                f" match up with <N_MFCCS> ({self.N_MFCCS})."
                )
        
        # Determine which coefficients to use
        # Select only MFCCs
        if self.form.lower() == 'mfcc': 
            self.X = self.X[:, :self.N_MFCCS]
            self.X = self.X[:, self.coefficients]
        
        # Select only MFCC deltas
        elif self.form.lower() == 'delta': 
            self.X = self.X[:, self.N_MFCCS:]
            self.X = self.X[:, self.coefficients]
        
        # Select both MFCCs and their deltas
        elif self.form.lower() == 'mfcc&delta':
            coefficients = np.concatenate( (self.coefficients, 
                                            self.coefficients + self.N_MFCCS) )
            self.X = self.X[:, coefficients]
        
        else:
            raise ValueError(f"TIDIGITS data form <{self.form}> is not known.")
        
        # If enabled, normalize the data (wrt time-axis only)
        # Do not normalize
        if isinstance(self.normalization, type(None)):
            pass

        # Normalize to obtain mean of 0 and standard deviation of 1
        elif self.normalization.lower() == 'mean':
            means = np.mean(self.X, axis=2)[:, :, None]
            stdvs = np.std(self.X, axis=2)[:, :, None]
            self.X = (self.X - means) / (stdvs + 1e-6)
        
        # Normalize to the range [-1, 1], where max(abs(values)) is 1
        elif (self.normalization.lower() == 'range'):
            maxs = torch.max(torch.abs(self.X), axis=2).values[:, :, None]
            self.X /= maxs
        
        # Limit the time-dimension of <X> according to given <max_duration>
        self.X = self.X[:, :, :self.shape[1]]

        # NOTE: hack
        self.variable_durations = self.df['n-timebins'].to_numpy()
        
        print(f"<X> has shape: {self.X.shape}")