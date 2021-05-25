import nest
nest.ResetKernel()

import numpy as np
import numpy.random as rd
import time, os, pickle

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import mlab, cm


from ep_functions import run_network, plot_planes


####################################################################################

# Deprivation parameters: effects of whisker deprivation. Interacting decrease of parvalbumin-positive interneurons excitability though increase of their threshold, together with recurrent potentiation of synapses from parvalbumin-positive to excitatory neurons, run on grid:

xi_thresholds = np.arange(0.,3.01,0.15)
zeta_EPs  = np.linspace(1.,1.5,len(xi_thresholds))

# Test:
#xi_thresholds = np.arange(0.,3.01,3.)
#zeta_EPs  = np.linspace(1.,1.5,len(xi_thresholds))


# Stimulation (input Poisson processes):
stim_lgn = 1000. # in Hz
stim_bkg = 1000. # in Hz


# Run network in baseline state, without whisker-deprivation induced changes
bl_re, bl_rp, bl_e, bl_p = run_network(stim_lgn, stim_bkg, md_params=[1.,1.,1.,1.], dTH=0.)


E,P = [],[] # Lists to save neuron class average rates
DE,DP = [],[] # Lists to save neuron class average rates normalized by baseline (plotted later on)


for xi_threshold in xi_thresholds:
    # Auxiliary lists to save data from second loop
    e,p = [],[]
    de,dp = [],[]
    for zeta_EP in zeta_EPs:

        re,rp, exc, pvl = run_network(stim_lgn, stim_bkg, md_params=[zeta_EP, 1., 1., 1.], dTH=xi_threshold)

        # Save raw rates
        e.append(re)
        p.append(rp)

        # Calculate rates normalized by baseline
        drr = []
        rates_dep = [re,rp]
        for ii, rrr in enumerate([bl_re, bl_rp]):
            if rrr!=0.: # precautionary condition if we run a network whose initial rate was zero
                dr = rates_dep[ii]/rrr
            else:
                print('Caution: Initial rate is zero')
                dr = 0.
            drr.append(dr)                    

        de.append(drr[0])
        dp.append(drr[1])

    # Add lines from second loop to overall list of lists for rates in the parameter plane:
    E.append(e)
    P.append(p)

    DE.append(de)
    DP.append(dp)
    # finished


# Dictionary for saving data (pickled)
ep_data = {}
ep_data['model_description'] = 'LIF network with a single subtype of interneurons; Simulation of interacting decrease of parvalbumin-positive interneurons excitability though increase of their threshold, together with recurrent potentiation of synapses from parvalbumin-positive to excitatory neurons, zeta_EP-xi_theta plane.'

ep_data['xi_thresholds'] = xi_thresholds
ep_data['zeta_EPs'] = zeta_EPs


ep_data['baseline_rates'] = [bl_re, bl_rp]

ep_data['Rate_Exc']=E
ep_data['Rate_PV'] =P

ep_data['Norm_Rate_Exc']=DE
ep_data['Norm_Rate_PV'] =DP


flname = 'data_ep-wd_somatosensory_coba-lif'
place = flname

fl = open(place, 'wb')
pickle.dump(ep_data, fl)
fl.close()


###########################################
# Plot plane from saved data:

DD = open('data_ep-wd_somatosensory_coba-lif', 'rb')
data = pickle.load(DD)

xi_thresholds  = np.round(data['xi_thresholds'],2)
zeta_EPs = np.round(data['zeta_EPs'],2)

DE = data['Norm_Rate_Exc']
DP = data['Norm_Rate_PV']


fig = plot_planes(DE, DP, zeta_EPs, xi_thresholds, case='somatosensory', limits=[0.,1.,8.])
plt.savefig('ep-wd_somatosensory.png')
