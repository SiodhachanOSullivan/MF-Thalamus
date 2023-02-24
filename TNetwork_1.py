from brian2 import *
prefs.codegen.target = "numpy"

start_scope()

DT=0.1 # time step
defaultclock.dt = DT*ms
N_inh = 500 # number of inhibitory neurons
N_exc = 500 # number of excitatory neurons

TotTime=1000 # Simulation duration (ms)
duration = TotTime*ms
tt = np.linspace(0,TotTime, int(TotTime/DT))


# Equations ----------------------------------------------------------------------------------
eqs='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn : siemens
dGsynE/dt = -GsynE/Tsyn : siemens
Is:ampere
Cm:farad
gl:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
Ee:volt
Ei:volt
Tsyn:second
'''

# Populations----------------------------------------------------------------------------------


# Population 1 [inhibitory] - RE - Reticular

b_inh = 10.*pA
G_inh = NeuronGroup(N_inh, eqs, threshold='v > -20*mV', reset='v = -55*mV; w += b_inh', refractory='5*ms', method='heun')
# init:
G_inh.v = -65.*mV
G_inh.w = 0.*pA
# synaptic parameters
G_inh.GsynI = 0.0*nS
G_inh.GsynE = 0.0*nS
G_inh.Ee = 0.*mV
G_inh.Ei = -80.*mV
G_inh.Tsyn = 5.*ms
# cell parameters
G_inh.Cm = 200.*pF
G_inh.gl = 10.*nS
G_inh.Vt = -45.*mV
G_inh.Dt = 2.5*mV
G_inh.tau_w = 200.*ms
G_inh.Is = 0.0*nA # external input
G_inh.El = -75.*mV
G_inh.a = 8.0*nS


# Population 2 [excitatory] - TC - Thalamocortical

b_exc = 10.*pA
G_exc = NeuronGroup(N_exc, eqs, threshold='v > -20.0*mV', reset='v = -50*mV; w += b_exc', refractory='5*ms',  method='heun')
# init
G_exc.v = -65.*mV
G_exc.w = 0.*pA
# synaptic parameters
G_exc.GsynI = 0.0*nS
G_exc.GsynE = 0.0*nS
G_exc.Ee = 0.*mV
G_exc.Ei = -80.*mV
G_exc.Tsyn = 5.*ms
# cell parameters
G_exc.Cm = 160.*pF
G_exc.gl = 10.*nS
G_exc.Vt = -50.*mV
G_exc.Dt = 4.5*mV
G_exc.tau_w = 200.*ms
G_exc.Is = 0.0*nA # ext inp
G_exc.El = -65.*mV # -55
G_exc.a = 0.*nS


# external drive--------------------------------------------------------------------------
ext_inp = 2*Hz
P_ed = PoissonGroup(8000, rates=ext_inp)

# var_P = TimedArray([4*Hz,4*Hz,4*Hz,4*Hz,8*Hz], duration/5)
# var_P = TimedArray(4/2*(1-np.cos(200*np.pi*tt))*Hz, defaultclock.dt)
# P_ed=PoissonGroup(8000, rates='var_P(t)')

var_STIM = TimedArray([0*Hz,0*Hz,10*Hz,0*Hz], duration/4)
STIM_ed=PoissonGroup(500, rates='var_STIM(t)')


# Network-----------------------------------------------------------------------------

# quantal increment in synaptic conductances:
Qpe = 1*nS # from P_ed to G_exc (p -> e)
Qpi = 4*nS
Qei = 4*nS
Qii = 1*nS
Qie = 6*nS

# probability of connection
prbC= 0.05

# create synapses
S_12 = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qie')
S_12.connect('i!=j',p=prbC)

S_11 = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qii')
S_11.connect('i!=j',p=prbC*6)

S_21 = Synapses(G_exc, G_inh, on_pre='GsynE_post+=Qei')
S_21.connect('i!=j',p=prbC)

S_ed_in = Synapses(P_ed, G_inh, on_pre='GsynE_post+=Qpi')
S_ed_in.connect(p=prbC)

S_ed_ex = Synapses(P_ed, G_exc, on_pre='GsynE_post+=Qpe')
S_ed_ex.connect(p=prbC*2)


S_st_ex = Synapses(STIM_ed, G_exc, on_pre='GsynE_post+=1*nS')
S_st_ex.connect(p=prbC*4)


# Recording tools -------------------------------------------------------------------------------

M1G_inh = SpikeMonitor(G_inh)
FRG_inh = PopulationRateMonitor(G_inh)
M1G_exc = SpikeMonitor(G_exc)
FRG_exc = PopulationRateMonitor(G_exc)

FRG_ed = PopulationRateMonitor(P_ed)


# Useful trick to record global variables ------------------------------------------------------

# Gw_inh = NeuronGroup(1, 'Wtot : ampere', method='rk4')
# Gw_exc = NeuronGroup(1, 'Wtot : ampere', method='rk4')

# SwInh1=Synapses(G_inh, Gw_inh, 'Wtot_post = w_pre : ampere (summed)')
# SwInh1.connect(p=1)
# SwExc1=Synapses(G_exc, Gw_exc, 'Wtot_post = w_pre : ampere (summed)')
# SwExc1.connect(p=1)

# MWinh = StateMonitor(Gw_inh, 'Wtot', record=0)
# MWexc = StateMonitor(Gw_exc, 'Wtot', record=0)



# GV_inh = NeuronGroup(1, 'Vtot : volt', method='rk4')
# GV_exc = NeuronGroup(1, 'Vtot : volt', method='rk4')

# SvInh1=Synapses(G_inh, GV_inh, 'Vtot_post = v_pre : volt (summed)')
# SvInh1.connect(p=1)
# SvExc1=Synapses(G_exc, GV_exc, 'Vtot_post = v_pre : volt (summed)')
# SvExc1.connect(p=1)

# MVinh = StateMonitor(GV_inh, 'Vtot', record=0)
# MVexc = StateMonitor(GV_exc, 'Vtot', record=0)


# Run simulation -------------------------------------------------------------------------------

print('--##Start simulation##--')
run(duration)
print(' --##End simulation##--')

# Plots -------------------------------------------------------------------------------


# prepare raster plot
RasG_inh = array([M1G_inh.t/ms, [i+N_exc for i in M1G_inh.i]])
RasG_exc = array([M1G_exc.t/ms, M1G_exc.i])


# binning and seting a frequency that is time dependant
def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)

BIN = 5 # Size of the time windows in ms
time_array = arange(int(TotTime/DT))*DT



LfrG_exc=array(FRG_exc.rate/Hz)
TimBinned,popRateG_exc=bin_array(time_array, BIN, time_array),bin_array(LfrG_exc, BIN, time_array)

LfrG_inh=array(FRG_inh.rate/Hz)
TimBinned,popRateG_inh=bin_array(time_array, BIN, time_array),bin_array(LfrG_inh, BIN, time_array)

LfrG_ed=array(FRG_ed.rate/Hz)
TimBinned,popRateG_ed=bin_array(time_array, BIN, time_array),bin_array(LfrG_ed, BIN, time_array)

meanRate_inh, meanRate_exc = np.mean(popRateG_inh), np.mean(popRateG_exc)



# create the figure

fig=figure(figsize=(8,12))
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212)

ax1.plot(RasG_inh[0], RasG_inh[1], ',r')
ax1.plot(RasG_exc[0], RasG_exc[1], ',g')

ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Neuron index')

ax2.plot(TimBinned,popRateG_inh, 'r')
ax2.axhline(meanRate_inh, c='r',ls='--', label=f'mean inh: {meanRate_inh:.2f}')
ax2.plot(TimBinned,popRateG_exc, 'g')
ax2.axhline(meanRate_exc, c='g',ls='--', label=f'mean exc: {meanRate_exc:.2f}')

ax2.plot(TimBinned,popRateG_ed, c='black')


# ax2.set_title(f'mean inh: {meanRate_inh:.2f} | mean exc: {meanRate_exc:.2f}')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Firing Rate (Hz)')
plt.legend()

name_fig='TNetwork_PLOT.png'
plt.savefig(name_fig)

# global variables
np.save('TNetwork_out_stim', np.vstack((popRateG_exc,popRateG_inh)))

# name_rates='FR_2pop_Reg_ext_'+str(ext_inp)+'.npy'
# np.save(name_rates,np.array([BIN, TimBinned, popRateG_inh,popRateG_exc,LfrG_inh,LfrG_exc], dtype=object))

# np.save('Wtot.npy',[np.array(MWinh.Wtot[0]/mamp),np.array(MWexc.Wtot[0]/mamp)])

# np.save('Vtot.npy',[np.array(MVinh.Vtot[0]/mV),np.array(MVexc.Vtot[0]/mV)])

# fig.tight_layout()

# show()
