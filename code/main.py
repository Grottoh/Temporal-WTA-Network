
# Make a data set iterator for data?

# Include the option to save and load neuron learning rates

# Create (integer) time constants with the smallest value being a microsecond
#  to allow for time between spikes (in particular lessen neurons in the input 
#  layer spiking at the same time).
# Also add delays times, e.g. for spikes travelling from one neuron to the 
#  next and for refractory periods. Ideally these delay times should be similar
#  for similar neurons, but slightly different. Also they delay should differ
#  between different spikes travelling the same synapse (i.e. add some 
#  randomness for these things). Perhaps delays could even be impacted by 
#  learning somehow.
# Not sure if I should make this a priority, and how far I should go with this.
#  Incorporating the above in my code would be nice, but might get complex, 
#  and might also slow the code down too might to be practical (at least in 
#  early stages of my research).

import os
import sys
import numpy as np
import torch
from system import Network

try:
    os.nice(19)
except:
    pass

import architectures    

if __name__ == "__main__":
    
    np.set_printoptions(precision=2, suppress=True)
    torch.set_printoptions(precision=2, sci_mode=False)
    
    # NEXT STEPS (one step at a time!):
    # 1) DONE add supervision evaluation during training
    # 2) DONE (first part) fix/improve saving and loading of models; perhaps allow loading of separate parts of models
    # 3) Add adaptive threshold, adaptive timescale, 
    #     and perhaps adaptive amount of neurons (by neuron death and neuro-genesis)
    # 4) Explore hierarchical design to reduce number of (synapse) parameters
    # 5) Perhaps find or design a different spike encoding of audio
    # 6) See whether network can 'remember' complex (audio) input patterns (i.e. train_data==test_data)
    # 7) See whether network can extrapolate this to unseen samples of the same class
    
    # Will need to address class imbalance at some point (e.g. take into 
    #   account that 90% accuracy is not good if 90% of the samples belong to one class, 
    #   and the model simply always predicts that class)
    
    # Saving and loading might still have problems (accuracy not the same after loading)
    
    # Seems one layer in second tier can also kindof learn repeating patterns?
    # n_layers=1
    # mp_max=100 
    # shape=(8, )
    # timescale=6
    # Achieved an accuracy of 92.40% in 1000 training iters
    # Anyway, a lot less complexity is needed than I initially thought,
    #  and I think an important step could be adding something that ensures
    #  neurons in separate circuits pay attention to different kinds of information
    
    
    # Make a Component parent class, and assign IDs to components (at least useful for printing and saving stuff)
    # Also make saved images have names such that alphanumerical order corresponds to chronological order
    # Evaluation can be a lot simpler and direct after supervised training
    
    # Many simultaneous spikes -> neuron death; every once in a while neuron born;
    #   if there are too many neurons, there will be many simultaneous spikes,
    #   this will balance to the appropriate amount of neurons
    
    # Include inhibitory connections/axons/synapses in more places
    
    # Give each component a 'global-run-info-item' which conveys information about e.g. whether we are training or testing?
    
    # Different system than tiers?
    # ids per component
    # saving and loading does not work properly right now I believe; 
    #   either way, make it so that components can be loaded into a network separately
    
    # Hierarchical design to reduce the number of synapses? 
    #   Possibly will require an increase in neurons (though maybe fewer connections as well?)
    
    # I need to allow for restoring (encapsulated) networks to different devices
    # Saving and loading not working properly atm anyway
    
    # z (axon)
    # self.spikes_t = torch.logical_not(self.X_i) # Only fire active neurons # ***** @@@@@ !!!!!!! (spikegenbin)
    

# =============================================================================
# if spiked: # *** !!! @@@ (inhibitor)
#     #if self.inh_strength < 0:
#     self.inh_strength *= 0.1
# =============================================================================
    

# =============================================================================
#             # *** !!! @@@ (axon learn)
#             self.weights[:, self.layer_b.spikes_t_a.to(torch.bool)] -= 0.0001 #* (-1 if self.inhibitory else 1)
#             self.weights = torch.clip(self.weights, min=self.w_min, max=self.w_max)
# =============================================================================


# =============================================================================
#         if self.XXX > 10000: (inhibitor)
#             return 0
#         else:
#             print(self.XXX)
#             self.XXX += 1
# =============================================================================


# =============================================================================
#         if self.inhibitory:
#             return weights # !!! *** @@@ (axon, normalize_weights)
# =============================================================================

# =============================================================================
#             dw = ys*torch.exp(-(weights-w_max - 4) / 2) - 1 # !!! @@@ *** (stdp)
# =============================================================================
            

# =============================================================================
#         # Might not work properly for <mode=='black-and-white'> !!! @@@ *** (layer SpikeGenBin)
#         max_value = torch.max((weight_pixels))
#         min_value = torch.min((weight_pixels))
#         weight_pixels = (weight_pixels - min_value) / max_value
# =============================================================================

    # TAKE CARE THAT IT ALSO RESETS STUFF FOR LATER LAYERS *** (neurogenesis)
    # MULTIPLE LAYERS CURRENTLY NOT COMPATIBLE WITH NEW TAG SYSTEM

    factor_monitor = 100
    factor_stimuli = 10
    iv_monitor = int(1*factor_monitor)
    en_save = True
    
# =============================================================================
#     # Create or load a network
# =============================================================================
    #network = Network.load(id_network='net-mnist-single_0000', i_run=-1, i_checkpoint=-1, pf_network="", change_pd_network=False, excl_extras=[])
    #network = Network.load(id_network='net-toy-single_0000', i_run=-1, i_checkpoint=-1, pf_network="", change_pd_network=False, excl_extras=[])
    #network = Network.load(id_network='net-toy-double_0000', i_run=-1, i_checkpoint=-1, pf_network="", change_pd_network=False, excl_extras=[])
    #network = Network.load(id_network='net-timit-double_0002', i_run=-1, i_checkpoint=-1, pf_network="", change_pd_network=False, excl_extras=[])
    #network = Network.load(id_network='', pf_network="[02LF] networks/networks-ponyland/net-timit-double_0001/run-00_trn50000/checkpoints/network-00_23000.pkl", change_pd_network=True, excl_extras=[])
        
    tiers, axons, ndt_network = architectures.get_mnist_single(iv_monitor, data_mode='no-black', en_layer_sv=True)
    #tiers, axons, ndt_network = architectures.get_static_test(iv_monitor)
    #tiers, axons, ndt_network = architectures.get_static_test2(iv_monitor)
    #tiers, axons, ndt_network = architectures.get_toy_single(iv_monitor)
    #tiers, axons, ndt_network = architectures.get_toy_double(iv_monitor)
    #tiers, axons, ndt_network = architectures.get_timit_double(iv_monitor)
    #tiers, axons, ndt_network = architectures.get_voice(iv_monitor)
    #tiers, axons, ndt_network = architectures.get_voice2(iv_monitor)
    #tiers, axons, ndt_network = architectures.get_voice3(iv_monitor)
    #tiers, axons, ndt_network = architectures.test(1); en_save = False
    network = Network(tiers=tiers, axons=axons, ndt_network=ndt_network)
    
# =============================================================================
#     # Train the network
# =============================================================================
    network.run(train=True , n_stimuli=int(factor_stimuli*factor_monitor), en_reset_on_new_stimulus=True, 
                en_evaluate=True, iv_evaluate=999500, n_evaluate=100, 
                en_save=en_save, en_save_network=True, iv_save_network=iv_monitor)
    
    try:
        if network.layers['LayerWTA'].en_neurogenesis:
            print("\nLooking to see which neurons should not be used:")
            for k in range(network.layers["LayerWTA"].mps.numel()):
                if network.layers["LayerWTA"].n_spikes_since_init[k] < 100:
                    network.layers["LayerWTA"].kill_neuron(k)
            print()
    except:
        pass
    
# =============================================================================
#     # Test the network
# =============================================================================
    factor_stimuli = 4
    try:    
        print("\nThresholds after training:")
        print(network.layers['LayerWTA'].thresholds)
        print()
    except:
        pass
    network.run(train=False, n_stimuli=int(factor_stimuli*factor_monitor), en_reset_on_new_stimulus=True, 
                en_save=en_save, en_save_network=False, iv_save_network=iv_monitor,
                test_with_train=True)
    network.run(train=False, n_stimuli=int(factor_stimuli*factor_monitor), en_reset_on_new_stimulus=True, 
                en_save=en_save, en_save_network=False, iv_save_network=iv_monitor,
                test_with_train=False)
    
    pass
