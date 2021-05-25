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
# Deprivation parameters: interacting feedforward synaptic depression as increase of feedforward E/I-ratio, together with recurrent potentiation of synapses from excitatory to parvalbumin-positive interneurons, run on grid:

delta_E = 1.# fixed depression of feedforward synapses onto excitatory neurons
delta_Ps=[] # calculate from feedforward E/I-ratios (rho_EIs)
zeta_PEs = np.arange(1.0,1.51,0.025)
rho_EIs  = np.arange(1.0,1.51,0.025)

#zeta_PEs = np.arange(1.0,1.51,0.5)
#rho_EIs  = np.arange(1.0,1.51,0.5)


# Stimulation (input Poisson processes):
stim_lgn = 1000. # in Hz
stim_bkg = 1000. # in Hz


# Run network in baseline state, without monocular-deprivation induced changes
bl_re, bl_rp, bl_e, bl_p = run_network(stim_lgn, stim_bkg, md_params=[1.,1.,1.,1.])


E,P = [],[] # Lists to save neuron class average rates
DE,DP = [],[] # Lists to save neuron class average rates normalized by baseline (plotted later on)


# Simulate rates in zeta_PE-rho_EI plane.
for rho_EI in rho_EIs:
    # Auxiliary lists to save date from second loop
    e,p = [],[]
    de,dp = [],[]
    for zeta_PE in zeta_PEs:

        # calculate depression of synapses onto parvalbumin-positive interneurons 
        delta_P = delta_E/rho_EI #change of feedforward E/I-ratio: delta_E/delta_P
        delta_Ps.append(delta_P)

        re,rp, exc, pvl = run_network(stim_lgn, stim_bkg, md_params=[1.,zeta_PE,delta_E,delta_P])

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
ep_data['model_description'] = 'LIF network with a single subtype of interneurons; Simulation of interacting feedforward synaptic depression as increase of feedforward E/I-ratio, together with recurrent potentiation following monocular deprivation, zeta_PE-rho_EI plane, multiplicative depression/potentiation.'

ep_data['delta_E'] = delta_E
ep_data['zeta_PEs'] = zeta_PEs
ep_data['delta_Ps'] = delta_Ps
ep_data['rho_EIs']  = rho_EIs


ep_data['baseline_rates'] = [bl_re, bl_rp]

ep_data['Rate_Exc']=E
ep_data['Rate_PV'] =P

ep_data['Norm_Rate_Exc']=DE
ep_data['Norm_Rate_PV'] =DP


flname = 'data_ep-md_interacting_coba-lif'
place = flname

fl = open(place, 'wb')
pickle.dump(ep_data, fl)
fl.close()


###########################################
# Plot plane from saved data:

DD = open('data_ep-md_interacting_coba-lif', 'rb')
data = pickle.load(DD)

rho_EIs  = np.round(data['rho_EIs'],2)
zeta_PEs = np.round(data['zeta_PEs'],2)

DE = data['Norm_Rate_Exc']
DP = data['Norm_Rate_PV']


fig = plot_planes(DE, DP, zeta_PEs, rho_EIs, case='interacting', limits=[0.,1.,5.])
plt.savefig('ep-md_interacting.png')
