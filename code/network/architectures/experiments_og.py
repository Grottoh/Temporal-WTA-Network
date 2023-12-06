import json
import os

import numpy as np
from network.architectures.tidigits.arch2l_tidigits import Arch2lTidigits

from network.architectures.toy.arch1l_toy import Arch1lToy
from network.layers.traits.neuron import NeuronSoftmax, NeuronStochastic
from network.probes.probe_experiment import ProbeExperiment
from network.spike_generators.sg_static import SGStatic
from network.spike_generators.sg_static_sequence import SGStaticSequence
from network.synapses.synapse import Synapse, SynapseDynamic

# Path to the directory where experiments are saved
PD_EXPERIMENTS = "../[03LF] experiments/"

def get_pd_experiment(ndt_experiment: str):
    """ 
    Experiment index is the number of already saved experiments of same name. 
    """
    idx_experiment = len(
        [name for name in os.listdir(PD_EXPERIMENTS) if 
            os.path.isdir(PD_EXPERIMENTS+name) and 
            name.split('_')[0]==ndt_experiment.split('_')[0]]
        )
    pd_experiment = PD_EXPERIMENTS + ndt_experiment.format(idx_experiment)
    return pd_experiment

def experiment_1a(
        seeds: set = set([(i+1)*100 for i in range(10)]),
        ndt_experiment: str = "experiment-1a_{:02d}/",
        ndt_network: str = "staticenc-staticwta_s{:04d}_{}/",
        device: str = 'cpu',
        ) -> None:
    """ Static encoding of temporal-toy data and static WTA circuit. """
    
    # Determine the path to the experiment directory
    pd_experiment = get_pd_experiment(ndt_experiment=ndt_experiment)

    # Create and run a new network for each seed
    for seed in sorted(set(seeds)):
        network = Arch1lToy._create(
            type_sg=SGStatic,
            type_neuron=NeuronStochastic,
            type_synapse=Synapse,

            pd_networks=pd_experiment,
            ndt_network=ndt_network.format(seed, '{:02d}'),
            
            device=device,
            seed=seed,
            )
        Arch1lToy.run_network(network=network)

def experiment_1b(
        seeds: set = set([(i+1)*100 for i in range(10)]),
        ndt_experiment: str = "experiment-1b_{:02d}/",
        ndt_network: str = "temporalenc-staticwta_s{:04d}_{}/",
        device: str = 'cpu',
        ) -> None:
    """ Temporal encoding of temporal-toy data and static WTA circuit. """
    
    # Determine the path to the experiment directory
    pd_experiment = get_pd_experiment(ndt_experiment=ndt_experiment)

    # Create and run a new network for each seed
    for seed in sorted(set(seeds)):
        network = Arch1lToy._create(
            type_sg=SGStaticSequence,
            type_neuron=NeuronSoftmax,
            type_synapse=Synapse,

            pd_networks=pd_experiment,
            ndt_network=ndt_network.format(seed, '{:02d}'),
            
            device=device,
            seed=seed,
            )
        Arch1lToy.run_network(network=network)

def experiment_2a(
        seeds: set = set([(i+1)*100 for i in range(10)]),
        ndt_experiment: str = "experiment-2a_{:02d}/",
        ndt_network: str = "staticenc-temporalwta_s{:04d}_{}/",
        device: str = 'cpu',
        ) -> None:
    """ Static encoding of temporal-toy data and temporal WTA circuit. """
    
    # Determine the path to the experiment directory
    pd_experiment = get_pd_experiment(ndt_experiment=ndt_experiment)

    # Create and run a new network for each seed
    for seed in sorted(set(seeds)):
        network = Arch1lToy._create(
            type_sg=SGStatic,
            type_neuron=NeuronStochastic,
            type_synapse=SynapseDynamic,

            mps_max=10_000,

            pd_networks=pd_experiment,
            ndt_network=ndt_network.format(seed, '{:02d}'),
            
            device=device,
            seed=seed,
            )
        Arch1lToy.run_network(network=network)

def experiment_2b(
        seeds: set = set([(i+1)*100 for i in range(10)]),
        ndt_experiment: str = "experiment-2b_{:02d}/",
        ndt_network: str = "temporalenc-temporalwta_s{:04d}_{}/",
        device: str = 'cpu',
        ) -> None:
    """ Temporal encoding of temporal-toy data and temporal WTA circuit. """
    
    # Determine the path to the experiment directory
    pd_experiment = get_pd_experiment(ndt_experiment=ndt_experiment)

    # Create and run a new network for each seed
    for seed in sorted(set(seeds)):
        network = Arch1lToy._create(
            type_sg=SGStaticSequence,
            type_neuron=NeuronSoftmax,
            type_synapse=SynapseDynamic,

            pd_networks=pd_experiment,
            ndt_network=ndt_network.format(seed, '{:02d}'),
            
            device=device,
            seed=seed,
            )
        Arch1lToy.run_network(network=network)

def experiment_3(experiment: str, device: str = 'cpu'):
    """ Single TIDIGITS speaker. """

    # Speakers of experiment 3 (first five by alphabetical order)
    BOYS = ['am', 'bb']#, 'ci', 'cw', 'da']
    GIRLS = ['aa', 'ab']#, 'as', 'bp', 'ch']
    MEN = ['ae', 'aj']#, 'al', 'aw', 'bd']
    WOMEN = ['ac', 'ag']#, 'ai', 'an', 'bh']
    ALL = ([('b', id_boy) for id_boy in BOYS] + 
           [('g', id_girl) for id_girl in GIRLS] + 
           [('m', id_man) for id_man in MEN] + 
           [('w', id_woman) for id_woman in WOMEN])
    
    # Determine the path to the experiment directory
    pd_experiment = get_pd_experiment(
        ndt_experiment=f"experiment-3{experiment}"+"_{:02d}/")

    if experiment.lower() == 'a':
        shuffle=False
        l2_K=22
        train_is_test=True
        label_by=['index']
    elif experiment.lower() == 'b':
        shuffle=True
        l2_K=22
        train_is_test=True
        label_by=['index']
    elif experiment.lower() == 'c':
        shuffle=False,
        l2_K=11
        train_is_test=True
        label_by=['utterance']
    elif experiment.lower() == 'd':
        shuffle=False
        l2_K=11
        train_is_test=False
        label_by=['utterance']

    for prefix, id_speaker in ALL:
        _experiment_3(
            l2_K=l2_K,
            shuffle=shuffle,
            label_by=label_by,
            train_is_test=train_is_test,
            id_speaker=id_speaker,
            pd_experiment=pd_experiment,
            ndt_network=f"{prefix}-{id_speaker}"+"_s{:04d}_{}/",
            device=device,
            )

def _experiment_3(
        l2_K: int,
        shuffle: bool,
        label_by: list[str],
        train_is_test: bool,
        id_speaker: str,
        pd_experiment: str,
        seeds: set = set([(i+1)*100 for i in range(10)]),
        ndt_network: str = "single-speaker_s{:04d}_{}/",
        device: str = 'cpu',
        ) -> None:
    """ Single TIDIGITS speaker. """

    # Number of unique training stimuli
    N_UNIQUE = 22

    # How many times to observe the entire train set
    N_CYCLES_TRN = 3
    #N_CYCLES_TRN = 50

    # How many times to observe the entire test set
    #N_CYCLES_TST = 1
    N_CYCLES_TST = 4

    # After how many stimuli layer-2 becomes active
    L2_IDLE_UNTIL = N_UNIQUE*2
    #L2_IDLE_UNTIL = N_UNIQUE*40

    #xxx = list(range(22)) if not train_is_test else []

    # Create and run a new network for each seed
    for seed in sorted(set(seeds)):
        network = Arch2lTidigits._create(
            l1_type_neuron=NeuronStochastic,
            
            label_by = label_by,
            whitelists=[{
                'utterance': ['o', 'z', '1', '2', '3', '4', 
                              '5', '6', '7', '8', '9'],
                'speaker-id': [id_speaker],
                }],
            shuffle=shuffle, # False<->experiment-a; True<->experiment-b

            l1_n_circuits=5,
            l1_K=(100,),
            l1_cst_idle=2,

            l2_K=(l2_K,),
            l2_idle_until=L2_IDLE_UNTIL,

            pd_networks=pd_experiment,
            ndt_network=ndt_network.format(seed, '{:02d}'),

            #eta_init=0.01,
            #eta_decay=0.60,
            #eta_star=25,

            en_visualize_weights=False,
            #en_visualize_weights=True,

            device=device,
            seed=seed,
            )
        Arch2lTidigits.run_network(
            network=network,

            #en_train=False,

            n_unique_trn=N_UNIQUE,
            n_cycles_trn=N_CYCLES_TRN,

            n_unique_tst=N_UNIQUE,
            n_unique_map=N_UNIQUE,
            n_cycles_tst=N_CYCLES_TST,

            data_subsets_trn=['train'],
            n_probings_trn=1,
            #n_probings_trn=66,

            data_subsets_map=['train'],
            data_subsets_tst=['train'],
            #n_probings_tst=2,
            n_probings_tst=1,
            )




NDS_EXPERIMENT = [
    "experiment-1a_{:02d}/", 
    "experiment-1b_{:02d}/",
    "experiment-2a_{:02d}/", 
    "experiment-2b_{:02d}/",
    ]


def evaluate_e1and2() -> None:
    
    for ndt_experiment in NDS_EXPERIMENT:
        nd_experiment = ndt_experiment.format(0)

        accuracies = dict()
        cnt_weights_total = None
        cnt_weights = dict()

        pd_experiment = PD_EXPERIMENTS+nd_experiment
        for nd_run in os.listdir(pd_experiment):
            pd_runs = pd_experiment + nd_run + '/'
            
            for nd_run in os.listdir(pd_runs):
                if 'test' in nd_run:
                    pf_results = pd_runs+nd_run+'/results.json'
                    with open(pf_results, 'r') as f:
                        results = json.load(f)

                        # ------------- Accuracy -------------
                        results_accuracy = results['LoggerEvaluate-layer-1']
                        for key, value in results_accuracy.items():
                            if key == 'accuracy':
                                continue
                            label = key.split('_')[1]
                            if not label in accuracies:
                                accuracies[label] = [float(value)]
                            else:
                                accuracies[label].append(float(value))
                        
                        # ------------- Weight counts -------------
                        if not 'LoggerWeights-weights_synapse' in results:
                            continue

                        results_weights=results['LoggerWeights-weights_synapse']

                        cnt_total = int(results_weights['total'])
                        if (cnt_weights_total != None and 
                            cnt_weights_total != cnt_total):
                            raise ValueError("<cnt_total> must be same for each run.")
                        cnt_weights_total = cnt_total

                        cnt_zero = int(results_weights['0.0'])
                        if not 'zero' in cnt_weights:
                            cnt_weights['zero'] = [cnt_zero]
                        else:
                            cnt_weights['zero'].append(cnt_zero)

                        cnt_nonzero = int(results_weights['nonzero'])
                        if not 'nonzero' in cnt_weights:
                            cnt_weights['nonzero'] = [cnt_nonzero]
                        else:
                            cnt_weights['nonzero'].append(cnt_nonzero)

                    break
        
        print(f"{nd_experiment} - accuracies:")
        for key, value in accuracies.items():
            n_runs = len(value)
            mean = np.mean(value)
            stdv = np.std(value)
            minv = np.min(value)
            maxv = np.max(value)
            print(f" > {key}:")
            print(f" >>>             n_runs = {n_runs}")
            print(f" >>>               mean = {mean:.2f}")
            print(f" >>> standard deviation = {stdv:.2f}")
            print(f" >>>            minimum = {minv:.2f}")
            print(f" >>>            maximum = {maxv:.2f}")

        print(f"{nd_experiment} - synapse weight counts:")
        print(f" > Total synapse weight count: {cnt_weights_total}")
        for key, value in cnt_weights.items():
            mean = np.mean(value)
            stdv = np.std(value)
            minv = np.min(value)
            maxv = np.max(value)
            print(f" > {key}:")
            print(f" >>>             n_runs = {n_runs}")
            print(f" >>>               mean = {mean:.2f}")
            print(f" >>> standard deviation = {stdv:.2f}")
            print(f" >>>            minimum = {minv:.2f}")
            print(f" >>>            maximum = {maxv:.2f}")
        
        print()
