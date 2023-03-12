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


meanRate_inh=[]
meanRate_exc=[]

Npts=20
ff=4. # in Hz
bb=10. # in pA
# frange=np.linspace(0.01,30,Npts)
# for step, ff in enumerate(frange):
brange=np.linspace(0,200,Npts)
for step, bb in enumerate(brange):

    print(f' {step+1}/{Npts}', end='\r')

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

    b_inh = bb*pA
    G_inh = NeuronGroup(N_inh, eqs, threshold='v > -20*mV', reset='v = -55*mV; w += b_inh', refractory='5*ms', method='heun')
    # init:
    G_inh.v = -55.*mV
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

    b_exc = bb*pA
    G_exc = NeuronGroup(N_exc, eqs, threshold='v > -20.0*mV', reset='v = -50*mV; w += b_exc', refractory='5*ms',  method='heun')
    # init
    G_exc.v = -50.*mV
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
    ext_inp = ff*Hz
    P_ed = PoissonGroup(8000, rates=ext_inp)

    # var_P = TimedArray([4*Hz,4*Hz,4*Hz,4*Hz,8*Hz], duration/5)
    # var_P = TimedArray(4/2*(1-np.cos(200*np.pi*tt))*Hz, defaultclock.dt)
    # P_ed=PoissonGroup(8000, rates='var_P(t)')

    # var_STIM = TimedArray([0*Hz,0*Hz,20*Hz,0*Hz], duration/4)
    # STIM_ed=PoissonGroup(500, rates='var_STIM(t)')


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


    # S_st_ex = Synapses(STIM_ed, G_exc, on_pre='GsynE_post+=1*nS')
    # S_st_ex.connect(p=prbC*4)


    # Recording tools -------------------------------------------------------------------------------

    M1G_inh = SpikeMonitor(G_inh)
    FRG_inh = PopulationRateMonitor(G_inh)
    M1G_exc = SpikeMonitor(G_exc)
    FRG_exc = PopulationRateMonitor(G_exc)

    # FRG_ed = PopulationRateMonitor(P_ed)



    # Run simulation -------------------------------------------------------------------------------

    # print('--##Start simulation##--')
    run(duration)
    # print(' --##End simulation##--')

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

    # LfrG_ed=array(FRG_ed.rate/Hz)
    # TimBinned,popRateG_ed=bin_array(time_array, BIN, time_array),bin_array(LfrG_ed, BIN, time_array)

    meanRate_inh.append(np.mean(popRateG_inh[int(len(popRateG_inh)/2):]))
    meanRate_exc.append(np.mean(popRateG_exc[int(len(popRateG_exc)/2):]))



# SAVE
np.save('data\\TNetwork_scan_b', np.vstack((meanRate_exc,meanRate_inh)))


# create the figure

plt.plot(brange, meanRate_exc, 'og')
plt.plot(brange, meanRate_inh, 'or')
plt.show()
# name_fig='TNetwork_scan_PLOT.png'
# plt.savefig(name_fig)

print()
