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

# Deprivation parameters: feedforward synaptic depression, run on grid:
#delta_Es = np.arange(0.5,1.01,0.5)
#delta_Ps = np.arange(0.5,1.01,0.5)

delta_Es = np.arange(0.5,1.01,0.025)
delta_Ps = np.arange(0.5,1.01,0.025)


# Stimulation (input Poisson processes):
stim_lgn = 1000. # in Hz
stim_bkg = 1000. # in Hz


# Run network in baseline state, without monocular-deprivation induced changes
bl_re, bl_rp, bl_e, bl_p = run_network(stim_lgn, stim_bkg, md_params=[1.,1.,1.,1.])


E,P = [],[] # Lists to save neuron class average rates
DE,DP = [],[] # Lists to save neuron class average rates normalized by baseline (plotted later on)

for delta_P in delta_Ps:
    # Auxiliary lists to save date from second loop
    e,p = [],[]
    de,dp = [],[]
    for delta_E in delta_Es:
        re,rp, exc, pvl = run_network(stim_lgn, stim_bkg, md_params=[1.,1.,delta_E,delta_P])

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
ep_data['model_description'] = 'LIF network with a single subtype of interneurons; Simulation of feedforward depression following monocular deprivation, delta_E-delta_P plane, multiplicative depression.'

ep_data['delta_Es'] = delta_Es
ep_data['delta_Ps'] = delta_Ps

ep_data['baseline_rates'] = [bl_re, bl_rp]

ep_data['Rate_Exc']=E
ep_data['Rate_PV'] =P

ep_data['Norm_Rate_Exc']=DE
ep_data['Norm_Rate_PV'] =DP


flname = 'data_ep-md_feedforward_coba-lif'
place = flname

fl = open(place, 'wb')
pickle.dump(ep_data, fl)
fl.close()


###########################################
# Plot plane from saved data:

DD = open('data_ep-md_feedforward_coba-lif', 'rb')
data = pickle.load(DD)

delta_Es = np.round(data['delta_Es'],2)
delta_Ps = np.round(data['delta_Ps'],2)

DE = data['Norm_Rate_Exc']
DP = data['Norm_Rate_PV']


fig = plot_planes(DE, DP, delta_Es, delta_Ps, case='feedforward', limits=[0.,1.,10.])
plt.savefig('ep-md_feedforward.png')
