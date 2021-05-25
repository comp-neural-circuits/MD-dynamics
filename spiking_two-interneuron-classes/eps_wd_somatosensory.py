import nest
nest.ResetKernel()

import numpy as np
import numpy.random as rd
import time, os, pickle

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import mlab, cm


from eps_functions import run_network, plot_planes


####################################################################################

# Deprivation parameters: effects of whisker deprivation. Interacting decrease of parvalbumin-positive interneurons excitability though increase of their threshold, together with recurrent potentiation of synapses from parvalbumin-positive to excitatory neurons, run on grid:

xi_thresholds = np.arange(0.,3.1,0.15)
zeta_EPs  = np.linspace(1.,1.5,len(xi_thresholds))

# Test:
#xi_thresholds = np.arange(0.,3.01,1.5)
#zeta_EPs  = np.linspace(1.,1.5,len(xi_thresholds))


# Stimulation (input Poisson processes):
stim_lgn = 1000. # in Hz
stim_bkg = 1000. # in Hz


# Run network in baseline state, without whisker-deprivation induced changes
bl_re, bl_rp, bl_rs, bl_e, bl_p, bl_s = run_network(stim_lgn, stim_bkg, md_params=[1.,1.,1.,1.], dTH=0.)


E,P,S = [],[],[] # Lists to save neuron class average rates
DE,DP,DS = [],[],[] # Lists to save neuron class average rates normalized by baseline (plotted later on)


for xi_threshold in xi_thresholds:
    # Auxiliary lists to save data from second loop
    e,p,s = [],[],[]
    de,dp,ds = [],[],[]
    for zeta_EP in zeta_EPs:

        re,rp,rs, exc, pvl, sst = run_network(stim_lgn, stim_bkg, md_params=[zeta_EP, 1., 1., 1.], dTH=xi_threshold)

        # Save raw rates
        e.append(re)
        p.append(rp)
        s.append(rs)

        # Calculate rates normalized by baseline
        drr = []
        rates_dep = [re,rp,rs]
        for ii, rrr in enumerate([bl_re, bl_rp, bl_rs]):
            if rrr!=0.: # precautionary condition if we run a network whose initial rate was zero
                dr = rates_dep[ii]/rrr
            else:
                print('Caution: Initial rate is zero')
                dr = 0.
            drr.append(dr)                    

        de.append(drr[0])
        dp.append(drr[1])
        ds.append(drr[2])

    # Add lines from second loop to overall list of lists for rates in the parameter plane:
    E.append(e)
    P.append(p)
    S.append(s)

    DE.append(de)
    DP.append(dp)
    DS.append(ds)
    # finished


# Dictionary for saving data (pickled)
eps_data = {}
eps_data['model_description'] = 'LIF network with two subtypes of interneurons; Simulation of interacting decrease of parvalbumin-positive interneurons excitability though increase of their threshold, together with recurrent potentiation of synapses from parvalbumin-positive to excitatory neurons, zeta_EP-xi_theta plane.'

eps_data['xi_thresholds'] = xi_thresholds
eps_data['zeta_EPs'] = zeta_EPs


eps_data['baseline_rates'] = [bl_re, bl_rp, bl_rs]

eps_data['Rate_Exc']=E
eps_data['Rate_PV'] =P
eps_data['Rate_SST']=S

eps_data['Norm_Rate_Exc']=DE
eps_data['Norm_Rate_PV'] =DP
eps_data['Norm_Rate_SST']=DS


flname = 'data_eps-wd_somatosensory_coba-lif'
place = flname

fl = open(place, 'wb')
pickle.dump(eps_data, fl)
fl.close()


###########################################
# Plot plane from saved data:

DD = open('data_eps-wd_somatosensory_coba-lif', 'rb')
data = pickle.load(DD)

xi_thresholds  = np.round(data['xi_thresholds'],2)
zeta_EPs = np.round(data['zeta_EPs'],2)

DE = data['Norm_Rate_Exc']
DP = data['Norm_Rate_PV']
DS = data['Norm_Rate_SST']


fig = plot_planes(DE, DP, DS, zeta_EPs, xi_thresholds, case='somatosensory', limits=[0.,1.,8.])
plt.savefig('eps-wd_somatosensory.png')
