from __future__ import annotations
from typing import Optional
import os
import json
import torch
import numpy as np
import pickle
import shutil
import time

from network.probes.logger_evaluate import LoggerEvaluate

from .utils.util import Constants as C
from .utils.util import is_jsonable
from network.data_loaders.data import Data
#from .probes.logger import Logger
from .component import Component, Trait
from .layers.layer_sensory import LayerSensory
from .architectures.arrangement import Arrangement
from .probes.probe import Probe
from .probes.logger import Logger

class Network(Component):

    # Path to the directory where networks are saved
    PD_NETWORKS = "../[02LF] networks/"

    # Directory name template of a run
    NDT_RUN = "run-{:02d}_{}{:05d}/" # {idx_run}_{mode_run}{n_stimuli}

    # The number of timesteps per second
    HZ_TIMESTEP = 1000

    # TODO: reassess the purpose of this
    # NOTE: <TEST> currently indicates mapping and testing on the same data,
    # <EVALUATE> indicates that a mapping was already made in a <MAP> run (on
    # different data); TODO: use better names
    # Define some run modes
    TRAIN = "train"
    TEST = "test"
    MAP = "map"
    EVALUATE = "evaluate"

    def __init__(
            self,
            nd_network: Optional[str] = None,
            en_variable_duration: bool = False,
            **kwargs,
            ) -> None :
        super().__init__(**kwargs)

        # Name of the directory where this specific network will be saved
        self.nd_network = nd_network

        # NOTE: hack - whether or not to allow variable duration stimuli
        self.en_variable_duration = en_variable_duration

        # Is to specify the manner in which network components are arranged
        self.arrangement: Optional[Arrangement] = None
    
    @classmethod
    def _create(cls, **kwargs) -> Network:

        network = cls(**kwargs)

        # Initialize network components
        network._init_components()

        # Initialize network components
        network._connect_components()

        # Set the traits of network components
        network._set_traits() 
        
        # # Initialize the generic network structure
        network._init_arrangement()

        # Track kwargs so that they can be logged
        network.kwargs = kwargs

        return network
    
    @property
    def components(self) -> list[Component]:
        """ Return a list of the main network components. """
        #return [node.component for node in self.arrangement.nodes]
        return self.arrangement.components
    
    def activate(self):
        """ Ensure entire network is active for the entirety of any run. """
        self.idle_until = 0
        self.idle_after = float('inf')
        [value.activate() for value in self.__dict__.values() if 
         issubclass(type(value), Component)]
    
    def get_active(self, idx_stimulus: int) -> list[Component]:
        """ Return list of all active network components. """
        return [component for component in self.components if 
                component.is_active(idx_stimulus=idx_stimulus)]
    
    def evaluate(
        self,
        data_subsets_map: list[str],
        data_subsets_eval: list[str],
        n_stimuli_map: int,
        n_stimuli_tst: int,
        **kwargs
        ) -> None:
        """ 
        First do a run in which neurons are mapped to a label based on their
        spiking behaviour. Follow this up with a run that uses this mapping to
        evaluate the performance of the network (typically done with separate
        datasets).
        """
        
        # Decide a mapping from neurons to labels
        self.run(
            mode_run=self.MAP, 
            data_subsets=data_subsets_map, 
            n_stimuli=n_stimuli_map,
            **kwargs
            )

        # Retrieve the evaluators which now have a neuron-to-label mapping
        self.evaluators: list[LoggerEvaluate] = [
            probe for probe in self.probes if type(probe) == LoggerEvaluate
            ]
        
        # Fix the neuron-to-label mapping decided in the <MAP> run
        for evaluator in self.evaluators:
            evaluator.fix_mapping()
            evaluator.loggers = [] # Remove loggers of the mapping run
        
        # Evaluate the network given the aformentioned mapping
        self.run(
            mode_run=self.EVALUATE, 
            data_subsets=data_subsets_eval, 
            n_stimuli=n_stimuli_tst,
            **kwargs
            )
    
    def run(
            self,
            mode_run: str,
            data_subsets: list[str],
            n_stimuli: int,
            en_learn: bool,

            en_show: bool,
            en_show_logs: bool,
            en_show_visuals: bool,

            en_save: bool,
            en_save_logs: bool,
            en_save_visuals: bool,
            en_save_network: bool,
            en_save_tensors: bool,

            iv_probes: Optional[int] = None,
            iv_save_network: Optional[int] = None,
            ) -> None:
        """ Prepare a run according to the given parameters, then start it. """
        
        # Determine some specific regarding the run
        self.mode_run = mode_run # Determine which subset of the data to load
        self.data_subsets = data_subsets # Which subset(s) of the data to load
        self.n_stimuli = n_stimuli # Determine how many stimuli to observe
        self.en_learn = en_learn # Determine whether or not to enable learning
        
        # Determine whether to show certain things
        self.en_show = en_show
        self.en_show_logs = en_show_logs and self.en_show
        self.en_show_visuals = en_show_visuals and self.en_show

        # Determine whether to save certain things
        self.en_save = en_save
        self.en_save_logs = en_save_logs and self.en_save
        self.en_save_visuals = en_save_visuals and self.en_save
        self.en_save_network = en_save_network and self.en_save
        self.en_save_tensors = en_save_tensors and self.en_save_network

        # If enabled, every <iv_probes> stimuli the network is probed
        self.iv_probes = (self.n_stimuli // 10 if 
                          iv_probes == None else 
                          iv_probes)
        
        # If enabled, every <iv_save_network> stimuli the network is saved
        self.iv_save_network = (self.n_stimuli // 10 if 
                                iv_save_network == None else 
                                iv_save_network )

        # Prepare the run: initialize probes, load appropriate data, ...
        self._prepare()

        # Start the run
        self._run()

    def _prepare(self) -> None:
        """ Prepare the run: initialize probes, load appropriate data, ... """
        
        # The network's main logger, logs and/or prints network info
        self.logger = Logger(
            loggers = [],
            en_local_logging=True,

            component=self,
            network=self,
            en_show=True,
            en_save=True,
            device=self.device,
            )
        
        # Ensure the name of each network component is unique
        for i, comp_a in enumerate(self.components[:-1]):

            # Make a list of all components with an identical name to <comp_a>
            identicals = []
            for comp_b in self.components[i+1:]:
                if comp_a.name == comp_b.name:
                    identicals.append(comp_b)

            # If there are identical names, distinguish them via an integer
            if len(identicals) > 0:
                for j, component in enumerate( [comp_a] + identicals ):
                    component.name += f'-{j:02d}'

        # Initialize probes that monitor network activity
        self._init_probes()

        # Load the appropriate data
        for root in self.arrangement.roots:
            
            # Ensure the root component is a sensory layer
            if not type(root) == LayerSensory:
                raise TypeError(f"Network root <{root}> must be of type" +
                                f" {LayerSensory}.")
            
            # Load the data corresponding to the sensory layer
            root.load_data(sets=self.data_subsets, n_stimuli=self.n_stimuli)

        # If enabled, initialize directories where network data is to be saved
        self._init_directories()

        # Log statistics regarding the data class distribution
        # TODO: move to a LayerSensory logger perhaps, or consider how to do
        # this when dealing with multiple root layers (not a big deal)
        for root in self.arrangement.roots:
            msg = f"Total number of samples: {root.data.n_samples}\n"
            for i in range(len(np.unique(root.data.T))):
                n_samples_i = (root.data.T==i).sum()
                msg += f" > {n_samples_i} samples where T=={i}"
                msg += f" ({100 * n_samples_i/root.data.n_samples:5.2f}%).\n"
            self.logger.produce(msg+"\n")

        # Prepare probe statistics for the new run
        for statistic in self.statistics.values():
            statistic.on_run()
        
        # Prepare each component for the new run
        for component in self.components:
            component.on_run()
    
    def _init_probes(self):
        """ Initialize probes that monitor network activity. """
        pass # Implemented by child class
    
    def _init_directories(self) -> None:
        """ Initialize directories where network data is to be saved. """
        
        # If enabled, create a directory in which to save network data
        if self.en_save_logs or self.en_save_visuals or self.en_save_network:

            # If it does not exist already, create the networks directory
            if not os.path.isdir(self.pd_networks):
                os.mkdir(self.pd_networks)
                print(f"\nCreated directory <{self.pd_networks}>.\n")

            # If it has not already, determine directory name of the network
            if self.nd_network == None:

                # Network index is the nr of saved networks of the same name
                self.idx_network = len(
                    [name for name in os.listdir(self.pd_networks) if 
                    os.path.isdir(self.pd_networks+name) and 
                    name.split('_')[0]==self.ndt_network.split('_')[0]]
                    )
                
                # Directory name only needs network index to be completed
                self.nd_network = self.ndt_network.format(self.idx_network)
            self.pd_network = self.pd_networks + self.nd_network # Net path
            
            # If it does not exist already, create the network directory
            if not os.path.isdir(self.pd_network):
                os.mkdir(self.pd_network)
                
            # Run index is the number of saved runs of this network
            self.idx_run = len(
                [name for name in os.listdir(self.pd_network) if 
                 os.path.isdir(self.pd_network+name) and 
                 name.split('-')[0]==self.NDT_RUN.split('-')[0]]
                )
            
            # Create directory that will contain data regarding upcoming run
            self.pd_run = self.pd_network + self.NDT_RUN.format(
                self.idx_run, 
                self.mode_run.lower(), 
                self.n_stimuli,
                )
            os.mkdir(self.pd_run)

            # Copy and save the code used for this run
            pd_codecopy = self.pd_run + C.ND_CODECOPY
            os.mkdir(pd_codecopy)
            for filename in os.listdir("./"):
                if filename.endswith('.py'):
                    shutil.copyfile(src=filename, dst=pd_codecopy+filename)
            shutil.copytree(src=C.ND_CODE, dst=pd_codecopy+C.ND_CODE)
            
            # Initialize the network's main logger
            self.logger.init_directories(
                pd_log=self.pd_run, 
                nf_log=self.nd_network[:-1]+".log")
            
            # Log non-default network settings
            if hasattr(self, 'kwargs'):
                self.logger.produce(f"Kwargs:\n{str(self.kwargs)}\n\n")
        
            if self.en_save_network:
                self.pd_checkpoints = self.pd_run+C.ND_CHECKPOINTS
                os.mkdir(self.pd_checkpoints)
                self.save(
                    en_save_tensors=self.en_save_tensors, 
                    ith_stimulus=None,
                    )

            # Create probe directories where necessary
            for probe in self.probes:
                probe.init_directories()
    
    def _run(self):
        """ Having prepared the run, start it. """

        # Observe a total of <self.n_stimuli> stimuli
        for idx_stimulus in range(self.n_stimuli):
            #start = time.time()
            ith_stimulus = idx_stimulus + 1 # Index of stimulus + 1

            # The number of timesteps for which to observe the current stimulus
            n_timesteps = max([root.S_i.shape[0] for 
                               root in self.arrangement.roots])

            # Prepare probe statistics for the new stimulus
            for statistic in self.statistics.values():
                statistic.on_start()
            
            # Perceive the current stimulus for <n_timesteps> timesteps
            finished = False
            for t in range(n_timesteps):

                # Update the state of each network component in proper order
                for component in self.get_active(idx_stimulus):
                    component.step()

                    # NOTE: hack - if enabled, end stim at variable duration
                    if self.en_variable_duration:
                        if hasattr(component, 'variable_duration'):
                            if component.variable_duration-1 == t:
                                finished = True
                        
                        if finished:
                            if hasattr(component, 'neuron'):
                                neuron = component.neuron
                                if hasattr(neuron, 'en_variable_duration'):
                                    if neuron.en_variable_duration:
                                        neuron.force_spike()
                
                for component in self.get_active(idx_stimulus):
                    component.step_two()
                
                # Update statistics for the current timestep
                for statistic in self.statistics.values():
                    statistic.step()
                
                # Where applicable have components learn from recent activity
                if self.en_learn: # Only learn if enabled ...
                    for component in self.get_active(idx_stimulus):
                        if component.is_learning(idx_stimulus):
                            component.learn()
                
                if self.en_learn:
                    for component in self.get_active(idx_stimulus):
                        if component.is_learning(idx_stimulus):
                            component.learn_two()
                
                # NOTE: hack - if enabled, end stim at variable duration
                if finished:
                    break

            # Update statistics for the completed stimulus
            for statistic in self.statistics.values():
                statistic.on_end()

            # Probe for information
            for probe in self.probes:
                probe.probe(
                    ith_stimulus=ith_stimulus, 
                    n_stimuli=self.n_stimuli,
                    )
            
            # Move on to the next stimulus
            for component in self.get_active(idx_stimulus):
                component.next()
            
            # If enabled, and the proper interval has passed, save the network
            if self.en_save_network: 
                if (ith_stimulus % self.iv_save_network == 0 or 
                    ith_stimulus == self.n_stimuli):
                        self.save(
                            en_save_tensors=self.en_save_tensors,
                            ith_stimulus=ith_stimulus,
                            )
                        
            #print(time.time()-start)
        
    @classmethod
    def run_network(
        cls,
        network: Network,
        
        en_train: bool,
        data_subsets_trn: list[str],
        n_unique_trn: int,
        n_cycles_trn: int,
        n_probings_trn: int,
        
        en_test: bool,
        data_subsets_map: list[str],
        data_subsets_tst: list[str],
        n_unique_map: int,
        n_unique_tst: int,
        n_cycles_tst: int,
        n_probings_tst: int,

        en_save: bool,
        en_save_logs: bool,
        en_save_visuals: bool,
        en_save_network: bool,
        en_save_tensors: bool,
        ) -> None:
        
        # Determine for how many stimuli to train and how often to probe
        n_stimuli_trn = n_unique_trn*n_cycles_trn
        iv_probes_trn = n_stimuli_trn // n_probings_trn
        
        # Determine how often to probe during testing
        n_stimuli_map = n_unique_map*n_cycles_tst
        n_stimuli_tst = n_unique_tst*n_cycles_tst
        iv_probes_tst = n_stimuli_tst // n_probings_tst
        
        # Train the network
        if en_train:
            
            # Do a run with train data
            network.run(
                mode_run=cls.TRAIN,
                data_subsets=data_subsets_trn,
                n_stimuli=n_stimuli_trn,
                en_learn=True,

                en_show=True,
                en_show_logs=True,
                en_show_visuals=True,

                en_save=en_save,
                en_save_logs=en_save_logs,
                en_save_visuals=en_save_visuals,
                en_save_network=en_save_network,
                en_save_tensors=en_save_tensors,

                iv_probes=iv_probes_trn,
                iv_save_network=float('inf'),
                )
            
        # Evaluate the network
        if en_test:
            
            # Ensure the network is active for the entirety of the test run
            network.activate()

            # Do a run with test data
            network.evaluate(
                data_subsets_map=data_subsets_map, 
                data_subsets_eval=data_subsets_tst,
                n_stimuli_map=n_stimuli_map,
                n_stimuli_tst=n_stimuli_tst,
                en_learn=False,

                en_show=True,
                en_show_logs=True,
                en_show_visuals=True,

                en_save=True,
                en_save_logs=True,
                en_save_visuals=True,
                en_save_network=True,
                en_save_tensors=en_save_tensors,

                iv_probes=iv_probes_tst,
                iv_save_network=float('inf'),
                )
    
    def save(
            self,
            en_save_tensors: bool = True, 
            ith_stimulus: Optional[int] = None,
            ) -> None:
        """ Save the network. """
        
        # Determine where to save the network, and create the directory
        nd_save = ("_checkpoint_init/" if 
                   ith_stimulus == None else 
                   f"checkpoint_{ith_stimulus:05d}/")
        pd_save = self.pd_checkpoints + nd_save
        os.mkdir(pd_save)

        # Save the class type of this network
        with open(pd_save+type(self).__name__+".pkl", 'wb') as f:
            pickle.dump(type(self), f)
        
        # Keep track of where the network was most recently saved
        self.pd_saved = pd_save

        # Put jsonables, tensors, arrays, components in separate dictionaries
        jsonables, tensors, arrays, components = dict(), dict(), dict(), dict()

        # Iterate over all network attributes that one might want to save
        for key, value in self.saveables.items():

            # Distinguish between saveables depending on how they are saved
            # Attribute must be saved via JSON
            if is_jsonable(value): 
                jsonables[key] = value

            # Attribute must be saved via torch
            elif type(value) == torch.Tensor:
                if en_save_tensors: # Only save torch tensors if enabled
                    tensors[key] = value
            
            # Attribute must be saved via numpy
            elif type(value) == np.ndarray:
                if en_save_tensors: # Only save numpy arrays if enabled
                    arrays[key] = value
            
            # Attribute must be saved as a component
            elif (issubclass(type(value), Component) and
                  not issubclass(type(value), Trait) and
                  not issubclass(type(value), Probe)):
                components[key] = value
            
            # Attribute is not saved
            else:
                pass
        
        # Save all JSON serializable attributes to a single .json file
        json_string = json.dumps(jsonables, indent=4)
        with open(pd_save+self.name+".json", "w") as f:
            f.write(json_string)
        
        # Save each torch tensor to a separate .pt file
        for key, value in tensors.items():
            torch.save(value, pd_save+key+".pt")
        
        # Save each numpy array to a separate .npy file
        for key, value in arrays.items():
            np.save(pd_save+key+".npy", value)
        
        # Save each network component to its own sub-directory
        for key, value in components.items():
            value.save(
                pd_save=pd_save,
                nd_component=key+"/",
                en_save_tensors=en_save_tensors,
                )
        
        # Log a message indicating when and where the network has been saved
        self.logger.produce(
            f"[{Logger.now()}] Saved the network to <{pd_save}>.\n\n"
        )

    
    @staticmethod
    def load(
        pd_load: str, 
        en_load_tensors: bool = True,
        ) -> Network:
        """ Load a network from directory <pd_load>. """
        
        # Load all saved network attributes
        type_network = None
        jsonables, tensors, arrays, components = dict(), dict(), dict(), dict()
        for filename in os.listdir(pd_load):

            # Load the class type of the network
            if filename.endswith(".pkl"):
                with open(pd_load+filename,'rb') as f:
                    type_network = pickle.load(f)
            
            # Load the attributes that were saved via JSON
            elif filename.endswith(".json"):
                with open(pd_load+filename, 'r') as f:
                    jsonables = json.load(f)
            
            # Load the attributes that were saved via torch
            elif filename.endswith(".pt") and en_load_tensors:
                key = filename.removesuffix(".pt")
                tensors[key] = torch.load(pd_load+filename)
            
            # Load the attributes that were saved via numpy
            elif filename.endswith(".npy") and en_load_tensors:
                key = filename.removesuffix(".npy")
                arrays[key] = np.load(pd_load+filename)
            
            # Load the attributes that were saved as network components
            elif os.path.isdir(pd_load+filename):
                key = filename
                components[key] = pd_load+filename+"/"
        
        # Initialize the network according to loaded JSON attributes
        network = type_network(**jsonables)

        # Set network tensors/arrays according to loaded torch/numpy attributes
        for key, value in (tensors|arrays).items():
            setattr(network, key, value)
        
        # Set network components according to the loaded network components
        for key, value in components.items():
            component = Component.load(
                pd_component=pd_load+key+"/",
                en_load_tensors=en_load_tensors,
                )
            setattr(network, key, component)
        
        # Connect the components of the network
        network._connect_components()

        # If tensors/arrays are not loaded, they will need to be initialized
        if not en_load_tensors:
            network._init_traits()
        
        # Initialize the network arrangement
        network._init_arrangement()

        # Keep track of where the network was loaded from
        network.pd_loaded = pd_load
        
        return network

    
    @staticmethod
    def load_by_ids(
        pd_network: str,
        idx_run: int = -1,
        idx_checkpoint: int = -1,
        en_load_tensors: bool = True,
        ) -> Network:
        """ Load a network. Indirectly specify path to network directory. """

        # Determine the path to the run directory
        pd_run = pd_network + sorted(
            os.listdir(pd_network)
            )[idx_run] + "/"
        
        # Determine the path to the checkpoint directory
        pd_checkpoints = pd_run + C.ND_CHECKPOINTS
        pd_checkpoint = pd_checkpoints + sorted( 
            os.listdir(pd_checkpoints)
            )[idx_checkpoint] + "/"
        
        return Network.load(
            pd_load=pd_checkpoint, 
            en_load_tensors=en_load_tensors,
            )

        
