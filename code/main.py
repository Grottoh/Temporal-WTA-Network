import os
from network.architectures.experiments.experiment_5 import Experiment5
from network.architectures.experiments.experiment_6 import Experiment6
from network.architectures.tidigits.arch1l_tidigits import Arch1lTidigits
from network.architectures.toy.arch1l_repeat import Arch1lRepeat
from network.architectures.toy.arch2l_repeat import Arch2lRepeat
from network.data_loaders.data_toy_repeat import DataToyRepeat

from network.network import Network
from network.architectures.mnist.mnist_single import MnistSingle
from network.architectures.toy.arch1l_toy import Arch1lToy
from network.architectures.tidigits.arch2l_tidigits import Arch2lTidigits

from network.data_loaders.data import Data
from network.data_loaders.data_mnist import DataMNIST
from network.data_loaders.data_toy_static import DataToyStatic
from network.data_loaders.data_toy_temporal import DataToyTemporal
from network.data_loaders.data_tidigits import DataTidigits

from network.architectures.experiments.experiment import Experiment
from network.architectures.experiments.experiment_1 import Experiment1
from network.architectures.experiments.experiment_2 import Experiment2
from network.architectures.experiments.experiment_3 import Experiment3
from network.architectures.experiments.experiment_4 import Experiment4

# TODO: ensure code can be run deterministically and is reproducible (cuda
# currently introduces some randomness)

# NOTE: Not sure if saving and loading of all networks work at the moment.

# Be nice when running on the servers
try:
    os.nice(19)
except:
    pass

if __name__ == "__main__":
    
    DataToyStatic.to_dataframe()
    # DataToyTemporal.to_dataframe()
    # DataToyRepeat.to_dataframe()
    # DataMNIST.to_dataframe()
    # DataTidigits.to_dataframe()

    Experiment1.run_1(idf='a', device='cpu')
    # Experiment1.run_2(idf='a', device='cpu')

    # Experiment2.run_1(idf='a', device='cpu')
    # Experiment2.run_2(idf='a', device='cpu')
    # Experiment2.run_3(idf='d', device='cpu')
    
    # Experiment3.run_1(idf='a', device='cpu')
    # Experiment3.run_2(idf='a', device='cpu')
    # Experiment3.run_3(idf='a', device='cpu')

    # Experiment4.run_1(idf='a', device='cuda')
    # Experiment4.run_2(idf='a', device='cuda')
    # Experiment4.run_3(idf='a', device='cuda')
    # Experiment4.run_4(idf='b', device='cuda')
    # Experiment4.run_5(idf='a', device='cuda')
    # Experiment4.run_6(idf='b', device='cuda')
    # Experiment4.run_6(idf='b', device='cuda')
    # Experiment4.run_7(idf='d', device='cpu')

    # Experiment5.run_1(idf='c', device='cuda')

    # Experiment6.run_1(idf='e', device='cpu')

    Experiment.evaluate(idf_experiment='1.1a_00')

    pass