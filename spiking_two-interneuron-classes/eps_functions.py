import nest
nest.ResetKernel()

import numpy as np
import numpy.random as rd
import time, os, pickle

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import mlab, cm


# Global parameters (not changeable in function)
dt = 0.1 # temporal resolution in ms
sim_time = 20.*1500.+300. # simulation time in ms
#sim_time = 20.*1500.+300. # simulation time in ms
tst = 300. # initial relaxation time

j_dev = 1.0 # weight for readout
j_st  = 0.5 # stimulus weight
delay = 0.1 # synaptic delay, 


NE = 4000 # number of excitatory neurons
NP = 1000 # number of parvalbumin-positive interneurons
NS = 500  # number of somatostatin-positive interneurons
NN = NE+NP+NS # total number of neurons

eps   = 0.1 # connection probability
n_e = int(eps*NE) # in-degree of excitatory neurons
n_p = int(eps*NP) # in-degree of parvalbumin-positive interneurons
n_s = int(eps*NS) # in-degree of somatostatin-positive interneurons



def run_network(stim_lgn, stim_bkg, J=0.1, K=1.6, g_rc=8., g_fw=2., md_params=[1.,1.,1.,1.], dTH=0.):
    '''
    Define structure for network with two subtypes of interneurons (parvalbumin- and somatostatin-positive) 
    and run with given parameters for monocular-deprivation induced synaptic changes.
    '''

    nest.ResetKernel()
    # Simulation parameters
    nest.SetStatus([0],{'grng_seed':1})
    nest.SetKernelStatus({"resolution": dt, 
                   "print_time": True, 
                   "overwrite_files": True,
                   "local_num_threads":5})

    # Parameters for leaky integrate-and-fire neurons with conductance based synapses
    neuron_params= {"C_m"    :  200.0,
                 "g_L"       :  10.0,
                 "tau_syn_ex":  5.,
                 "tau_syn_in":  5.,
                 "t_ref"     :  2.,
                 "V_m"       : -70.0,
                 "E_L"       : -70.0,
                 "V_th"      : -50.0,
                 "V_reset"   : -58.0,
                 "E_ex"      :  0.0,
                 "E_in"      : -85.0,
                 "I_e"       :  0.}
    nest.SetDefaults("iaf_cond_exp", neuron_params)
    

    print("Creating network")
    # Define nodes, inputs and devices:
    nodes_e = nest.Create("iaf_cond_exp", NE)
    nodes_p = nest.Create("iaf_cond_exp", NP)
    nodes_s = nest.Create("iaf_cond_exp", NS)
    nodes_all = nodes_e + nodes_p + nodes_s

    # Decrease PV intrinsic excitability through increase of threshold (model effects of whisker deprivation)
    nest.SetStatus(nodes_p, {"V_th":-50.0+dTH})

    # initialize with random membrane potentials for all neurons
    mems = 30.*rd.rand(NN) - 70.01
    nest.SetStatus(nodes_all,"V_m",mems)


    input_device_lgn = nest.Create("poisson_generator")
    input_device_bkg = nest.Create("poisson_generator")

    sp_e = nest.Create("spike_detector")
    sp_s = nest.Create("spike_detector")
    sp_p = nest.Create("spike_detector")

    nest.SetStatus(sp_e, 
        {"label":"spikes-ex", 
        "withgid":True, 
        "withtime":True,
        "to_file":False,
        "to_memory":True,
        "start": tst,
        "stop": np.inf})

    nest.SetStatus(sp_p, 
        {"label":"spikes-pv", 
        "withgid":True, 
        "withtime":True,
        "to_file":False,
        "to_memory":True,
        "start": tst,
        "stop": np.inf})

    nest.SetStatus(sp_s, 
        {"label":"spikes-sst", 
        "withgid":True, 
        "withtime":True,
        "to_file":False,
        "to_memory":True,
        "start": tst,
        "stop": np.inf})


    print("Connecting")
    # Define statistical properties of connectivity
    conn_dict_ffw  ={'rule': 'all_to_all'}

    conn_dict_e  ={'rule': 'fixed_indegree', 'indegree': n_e, 'autapses':False, 
                    'multapses':False}
    conn_dict_s  ={'rule': 'fixed_indegree', 'indegree': n_s, 'autapses':False, 
                   'multapses':False}
    conn_dict_p  ={'rule': 'fixed_indegree', 'indegree': n_p, 'autapses':False, 
                   'multapses':False}

    # Extract parameters for monocular-deprivation induced synaptic changes
    zeta_EP = md_params[0]
    zeta_PE = md_params[1]
    delta_E = md_params[2]
    delta_P = md_params[3]

    # Define cell type specific weights
    J_ETC = delta_E*j_st
    J_PTC = delta_P*g_fw*j_st
    J_BKG = j_st

    J_EE = J
    J_PE = zeta_PE*J
    J_SE = J

    J_EP = -zeta_EP*g_rc*J
    J_PP = -g_rc*J
    J_SP = 0.

    J_ES = -K
    J_PS = -K
    J_SS = 0.

    # Synapses for external drives
    syn_dict_etc = { "model":"static_synapse", "weight":J_ETC,"delay":0.1}
    syn_dict_ptc = { "model":"static_synapse", "weight":J_PTC,"delay":0.1}
    syn_dict_bkg = { "model":"static_synapse", "weight":J_BKG,"delay":0.1}
    syn_dict_det = { "model":"static_synapse", "weight":1.,"delay":0.1}

    # Recurrent synapses
    syn_dict_ee = { "model":"static_synapse", "weight":J_EE,"delay": delay}
    syn_dict_pe = { "model":"static_synapse", "weight":J_PE,"delay": delay}
    syn_dict_se = { "model":"static_synapse", "weight":J_SE,"delay": delay}

    syn_dict_ep = { "model":"static_synapse", "weight":J_EP,"delay": delay}
    syn_dict_pp = { "model":"static_synapse", "weight":J_PP,"delay": delay}
    syn_dict_sp = { "model":"static_synapse", "weight":J_SP,"delay": delay}

    syn_dict_es = { "model":"static_synapse", "weight":J_ES,"delay": delay}
    syn_dict_ps = { "model":"static_synapse", "weight":J_PS,"delay": delay}
    syn_dict_ss = { "model":"static_synapse", "weight":J_SS,"delay": delay}

    # Connect network:
    # External inputs
    nest.Connect(input_device_lgn, nodes_e, conn_dict_ffw, syn_dict_etc)
    nest.Connect(input_device_lgn, nodes_p, conn_dict_ffw, syn_dict_ptc)

    nest.Connect(input_device_bkg,  nodes_e, conn_dict_ffw, syn_dict_bkg)
    nest.Connect(input_device_bkg,  nodes_s, conn_dict_ffw, syn_dict_bkg)

    # Recurrent connections
    nest.Connect(nodes_e,nodes_e, conn_dict_e, syn_dict_ee)
    nest.Connect(nodes_e,nodes_p, conn_dict_e, syn_dict_pe)
    nest.Connect(nodes_e,nodes_s, conn_dict_e, syn_dict_se)

    nest.Connect(nodes_p,nodes_e, conn_dict_p, syn_dict_ep)
    nest.Connect(nodes_p,nodes_p, conn_dict_p, syn_dict_pp)
    nest.Connect(nodes_p,nodes_s, conn_dict_p, syn_dict_sp)

    nest.Connect(nodes_s,nodes_e, conn_dict_s, syn_dict_es)
    nest.Connect(nodes_s,nodes_p, conn_dict_s, syn_dict_ps)
    nest.Connect(nodes_s,nodes_s, conn_dict_s, syn_dict_ss)

    # Spike readout:
    nest.Connect(nodes_e, sp_e, conn_dict_ffw, syn_dict_det)
    nest.Connect(nodes_p, sp_p, conn_dict_ffw, syn_dict_det)
    nest.Connect(nodes_s, sp_s, conn_dict_ffw, syn_dict_det)


    print("Simulating")
    nest.ResetNetwork()
    nest.SetStatus(input_device_lgn, {"rate":stim_lgn})
    nest.SetStatus(input_device_bkg, {"rate":stim_bkg})


    # Simulate network (with time shown in terminal
    ti = time.time()
    nest.Simulate(sim_time)

    # Results
    # Readout number of spikes and calculate rates:
    spikes_e = nest.GetStatus(sp_e, 'n_events')[0]
    spikes_p = nest.GetStatus(sp_p, 'n_events')[0]
    spikes_s = nest.GetStatus(sp_s, 'n_events')[0]

    rate_e = spikes_e / (NE*(sim_time-tst)/1000.)
    rate_p = spikes_p / (NP*(sim_time-tst)/1000.)
    rate_s = spikes_s / (NS*(sim_time-tst)/1000.)


    # Readout spike times
    senders_e = nest.GetStatus(sp_e)[0]['events']['senders']
    times_e   = nest.GetStatus(sp_e)[0]['events']['times']

    senders_p = nest.GetStatus(sp_p)[0]['events']['senders']
    times_p   = nest.GetStatus(sp_p)[0]['events']['times']

    senders_s = nest.GetStatus(sp_s)[0]['events']['senders']
    times_s   = nest.GetStatus(sp_s)[0]['events']['times']

    ts=time.time()

    print("-------------------------------------------")
    print('MD-parameters='+str(md_params))
    print('PV-threshold change=%.1f'%(dTH))
    print("J, K = %.2f, %.2f"%(J, K))
    print("-------------------------------------------")
    print("Simulation time: %.2f s" % (ts-ti))
    print("LGN Stimulus : %.2f spikes/sec" %(stim_lgn))
    print("BKG Stimulus : %.2f spikes/sec" %(stim_bkg))
    print("Mean EXC rate: %.2f spikes/sec" % (rate_e))
    print("Mean PVL rate: %.2f spikes/sec" % (rate_p))
    print("Mean SST rate: %.2f spikes/sec" % (rate_s))
    print("------------------------------------------")

    # Return dictionary of spiketimes and emitting neurons
    exc = {'times':times_e, 'senders':senders_e}
    pvl = {'times':times_p, 'senders':senders_p}
    sst = {'times':times_s, 'senders':senders_s}

    return rate_e, rate_p, rate_s, exc, pvl, sst




##################################################################
##################################################################
def plot_planes(values_E, values_P, values_S, xcoords, ycoords, case='feedforward',colors='RdBu_r', limits=[None,None,None]):
    '''
    Plot heatmaps for normalized response in parameter planes for different cases of synaptic changes looked at (feedforward depression, recurrent potentiation or interacting feedforward depression and recurrent potentiation.
    case is given to the function as string: 'feedforward', 'recurrent', 'interacting' (all monocular deprivation) or 'somatosensory' (for whisker deprivation);
    '''
    jumpx = int(len(xcoords)/2)
    jumpy = int(len(ycoords)/2)

    # Limits and middle for heatmaps:
    VMN=limits[0]
    mid=limits[1]
    VMX=limits[2]

    # Extend x- and y-values to make proper grid
    xc = np.copy(xcoords)
    yc = np.copy(ycoords)
    xx, yy = np.meshgrid(xc,yc)

    # Colour normalisation
    class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))



    # Define levels for E, PV and SST (the same across classes but different ones chosen for inline notes)
    # Different levels for different cases (feedforward, recurrent and interacting feedforward and recurrent):
    if case=='feedforward':
        # Excitatory
        ll1e = np.array((0.25, 1., 2., 8.))#levels w/ inline notes
        ll2e = np.array((0., 0.1, 0.5, 0.75, 3.5, 10.))#levels w/o inline notes
        # Parvalbumin-positive
        ll1p = np.array((0.1, 0.5, 1., 2., 8.))#levels w/ inline notes
        ll2p = np.array((0., 0.25, 0.75, 3.5, 10.))#levels w/o inline notes
        # Somatostatin-positive
        ll1s = np.array((0.1, 1., 8.))#levels w/ inline notes
        ll2s = np.array((0., 0.25, 0.5, 0.75, 2., 3.5, 10.))#levels w/o inline notes
        
        # Also, choose x- and y-labels case dependent:
        x_lbl=r'$\delta_{E}$'
        y_lbl=r'$\delta_{P}$'

    elif case=='recurrent':
        # Excitatory
        ll1e = np.array((0.5, 1.,1.15, 1.3))#levels w/ inline notes
        ll2e = np.array((0., 0.1, 0.25, 0.75, 1.1,1.2, 1.25))#levels w/o inline notes
        # Parvalbumin-positive
        ll1p = np.array((0.1, 0.5, 1.,1.15, 1.3))#levels w/ inline notes
        ll2p = np.array((0., 0.25, 0.75,1.1,1.2,1.25))#levels w/o inline notes
        # Somatostatin-positive
        ll1s = np.array((0.1, 1.,1.15))#levels w/ inline notes
        ll2s = np.array((0., 0.25, 0.5, 0.75,1.1,1.2, 1.25, 1.3))#levels w/o inline notes

        # Also, choose x- and y-labels case dependent:
        x_lbl=r'$\zeta_{EP}$'
        y_lbl=r'$\zeta_{PE}$'

    elif case=='interacting':
        # Excitatory
        ll1e = np.array((0.25,1.,1.5,3.5))#levels w/ inline notes
        ll2e = np.array((0., 0.4, 0.7, 0.9, 1.2, 2.5, 7.,11.))#levels w/o inline notes
        # Parvalbumin-positive
        ll1p = np.array((0.4,1.,1.5,3.5))#levels w/ inline notes
        ll2p = np.array((0., 0.25, 0.7, 0.9, 1.2, 2.5, 7.,11.))#levels w/o inline notes
        # Somatostatin-positive
        ll1s = np.array((0.25,1.,7.))#levels w/ inline notes
        ll2s = np.array((0., 0.4, 0.7, 0.9, 1.2, 1.5, 2.5, 3.5,11.))#levels w/o inline notes

        # Also, choose x- and y-labels case dependent:
        x_lbl=r'$\zeta_{PE}$'
        y_lbl=r'$\rho_{EI}$'


    elif case=='somatosensory':
        # Excitatory
        ll1e = np.array((0.25, 0.4, 1., 1.5))#levels w/ inline notes
        ll2e = np.array((0., 0.7, 0.9, 1.2, 5., 7.))#levels w/o inline notes
        # Parvalbumin-positive
        ll1p = np.array((0.25, 0.4, 1.,1.5))#levels w/ inline notes
        ll2p = np.array((0., 0.7, 0.9, 1.2, 5., 7.))#levels w/o inline notes
        # Somatostatin-positive
        ll1s = np.array((0.25,1.,5.))#levels w/ inline notes
        ll2s = np.array((0., 0.4, 0.7, 0.9, 1.2, 1.5, 7.))#levels w/o inline notes

        # Also, choose x- and y-labels case dependent:
        x_lbl=r'$\zeta_{EP}$'
        y_lbl=r'$\xi_{\theta}$'

    else:
        print('Error: unknown case for plane of monocular-deprivation induced plasticity')

    # Put all together and sort for colormap levels, same accross neurons classes
    lls = np.sort(np.concatenate((ll1e,ll2e)))


    # Plot
    FS = 12
    FS1 = 9

    k = max(xx[0])-min(xx[0])
    current_cmap = plt.cm.get_cmap(colors)


    fig=plt.figure(figsize=((14./2.54),(5./2.54)))

    # Plot excitatory (heatmap & contours, with & without inline notes):
    ax1=plt.axes([0.06,0.16,0.25,0.7])
    plt.title('Exc',fontsize=FS)
    norm1 = MidpointNormalize(midpoint=mid)

    ime1 = ax1.contourf(xc,yc,values_E, cmap=cm.get_cmap(current_cmap, len(lls)), norm=norm1, vmin=VMN, vmax=VMX, levels=lls, extend="both")
    ime2 = ax1.contour(xc,yc,values_E,levels=ll1e, colors='black', linewidths=1.,vmax=VMX, vmin=VMN)
    plt.clabel(ime2, inline=1, fontsize=FS1,fmt='%.2f')
    ime3 = ax1.contour(xc,yc,values_E, levels=ll2e, colors='black', linewidths=1., vmax=VMX, vmin=VMN)


    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(0)

    ax1.xaxis.set_label_coords(0.5,-0.05)
    ax1.yaxis.set_label_coords(-0.05,0.5)

    ax1.set(xlabel=x_lbl, ylabel=y_lbl, 
            xticks=(xc[0],xc[-1]), xticklabels=(np.round([xc[0],xc[-1]],1)), 
            yticks=([yc[0],yc[-1]]), yticklabels=(np.round([yc[0],yc[-1]],1)))



    ##########################################
    # Plot PV normalized responses:
    ax2=plt.axes([0.35,0.16,0.25,0.7])

    plt.title('PV',fontsize=FS)
    norm2 = MidpointNormalize(midpoint=mid)

    imp1 = ax2.contourf(xc,yc,values_P, cmap=cm.get_cmap(current_cmap, len(lls)), norm=norm2, vmin=VMN, vmax=VMX, levels=lls, extend="both")
    imp2 = ax2.contour(xc,yc,values_P,levels=ll1p, colors='black', linewidths=1.,vmax=VMX, vmin=VMN)
    plt.clabel(imp2, inline=1, fontsize=FS1,fmt='%.2f')
    imp3 = ax2.contour(xc,yc,values_P, levels=ll2p, colors='black', linewidths=1., vmax=VMX, vmin=VMN)

    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(0)
    ax2.xaxis.set_label_coords(0.5,-0.05)
    ax2.yaxis.set_label_coords(-0.05,0.5)
    ax2.set(xlabel=x_lbl, #ylabel=y_lbl, 
            xticks=(xc[0],xc[-1]), xticklabels=(np.round([xc[0],xc[-1]],1)), 
            yticks=(), yticklabels=())



    ##########################################
    # Plot SST normalized responses:
    ax3=plt.axes([0.64,0.16,0.25,0.7])
    plt.title('SST',fontsize=FS)
    norm3 = MidpointNormalize(midpoint=mid)
    imp1 = ax3.contourf(xc,yc,values_S, cmap=cm.get_cmap(current_cmap, len(lls)), norm=norm2, vmin=VMN, vmax=VMX, levels=lls, extend="both")
    imp2 = ax3.contour(xc,yc,values_S,levels=ll1s, colors='black', linewidths=1.,vmax=VMX, vmin=VMN)
    plt.clabel(imp2, inline=1, fontsize=FS1,fmt='%.2f')
    imp3 = ax3.contour(xc,yc,values_S, levels=ll2s, colors='black', linewidths=1., vmax=VMX, vmin=VMN)

    for label in ax3.xaxis.get_ticklabels():
        label.set_rotation(0)
    ax3.xaxis.set_label_coords(0.5,-0.05)
    ax3.yaxis.set_label_coords(-0.05,0.5)
    ax3.set(xlabel=x_lbl, #ylabel=y_lbl, 
            xticks=(xc[0],xc[-1]), xticklabels=(np.round([xc[0],xc[-1]],1)), 
            yticks=(), yticklabels=())


    ##########################################
    # Shared colorbar 
    cbar_ax = fig.add_axes([0.91,0.15,0.02,0.7])
    cbar = plt.colorbar(imp1,cax=cbar_ax, ticks=[VMN,mid,VMX])
    cbar.ax.set_yticklabels(['%.1f'%(VMN),'%.1f'%(mid),'%.1f'%(VMX)],fontsize=FS1) 
    plt.show()


    return fig
