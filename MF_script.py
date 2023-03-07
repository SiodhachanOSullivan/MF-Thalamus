# Model from  Di Volo et al. Neural Comp. 2019


import numpy as np
import matplotlib.pylab as plt
from scipy.special import erfc
from mytools import progressBar, double_gaussian


def TF(typ,fexc,finh,adapt):

    if typ=='TC':
        P = PTC
        Nexc=800
        Ninh=25
        Qe=1e-9
        Qi=6e-9
        Cm=160e-12
        El=-65e-3
    elif typ=='RE':
        P = PRE
        Nexc=400
        Ninh=150
        Qe=4e-9
        Qi=1e-9
        Cm=200e-12
        El=-75e-3

    if fexc<1e-8: fe=1e-8
    else: fe = fexc*Nexc
    if finh<1e-8: fi=1e-8
    else: fi = finh*Ninh

    # fe+=1e-9;
    # fi+=1e-9;
    
    muGi = Qi*Ti*fi;
    muGe = Qe*Te*fe;
    muG = Gl+muGe+muGi;
    muV = (muGe*Ee+muGi*Ei+Gl*El-adapt)/muG;
    # muV = (muGe*Ee+muGi*Ei+Gl*El - fout*Tw*b + a*El)/(muG+a);
    
    
    muGn = muG/Gl;
    Tm = Cm/muG;
    
    Ue =  Qe/muG*(Ee-muV);
    Ui = Qi/muG*(Ei-muV);
    
    sV = np.sqrt(fe*(Ue*Te)*(Ue*Te)/2./(Te+Tm)+fi*(Ui*Ti)*(Ui*Ti)/2./(Ti+Tm));
    if sV<1e-4: sV=1e-4

    Tv = ( fe*(Ue*Te)*(Ue*Te) + fi*(Qi*Ui)*(Qi*Ui)) /( fe*(Ue*Te)*(Ue*Te)/(Te+Tm) + fi*(Qi*Ui)*(Qi*Ui)/(Ti+Tm) );
    TvN = Tv*Gl/Cm;
    
    muV0=-60e-3;
    DmuV0=10e-3;
    sV0=4e-3;
    DsV0=6e-3;
    TvN0=0.5;
    DTvN0=1.;

    vthre = P[0] + P[1]*(muV-muV0)/DmuV0 + P[2]*(sV-sV0)/DsV0 + P[3]*(TvN-TvN0)/DTvN0 \
        + P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0) + P[5]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0 + P[6]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0 + P[7]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0) + P[8]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0 + P[9]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0);


    frout = 1/(2*Tv) * erfc( (vthre - muV)/(np.sqrt(2)*sV) )
    if frout<1e-8: frout=1e-8
    
    return frout;


def MFw(typ, w, fexc, finh):
    if typ=='TC':
        adapt = -w/Tw+b*fexc
    if typ=='RE':
        a=8e-9
        Nexc=400
        Ninh=150
        Qe=4e-9
        Qi=1e-9
        El=-75e-3

        fe=fexc*Nexc
        fi=finh*Ninh
        
        muGi = Qi*Ti*fi;
        muGe = Qe*Te*fe;
        muG = Gl+muGe+muGi;
        muV = (muGe*Ee+muGi*Ei+Gl*El-w)/muG;
        adapt = -w/Tw+b*finh + a*(muV-El)/Tw
        
    return adapt



# constants
Gl=10*1.e-9
Tw=200*1e-3
b=10*1e-12

Ti=5*1.e-3 # Tsyn RE
Te=5*1.e-3 # Tsyn TC
Ee=0
Ei=-80*1.e-3


#-load fit params
PTC=np.load('data\\NEW2params_TC.npy')
PRE=np.load('data\\NEW2params_RE.npy')
# PTC=np.load("C:\VSCode\DB_comparison\mf\data\FS-cell_CONFIG1_fit.npy")[[0,1,2,3,5,8,9,6,10,7]]
# PRE=np.load("C:\VSCode\DB_comparison\mf\data\RS-cell_CONFIG1_fit.npy")[[0,1,2,3,5,8,9,6,10,7]]


#=MF and numerics params
T=10e-3

tfinal=2 # s
dt=5e-4 # s
df=1e-5 # Hz
tsteps=int(tfinal/dt)

t = np.linspace(0, tfinal, tsteps)


#=CORTEX external input
#---constant
external_input=np.full(tsteps, 4.)
# external_input+=np.random.randn(tsteps)/2

#-timeframe
# external_input=np.zeros(tsteps)
# external_input[:1000] = 4
# external_input[int(tsteps*2/4):int(tsteps*3/4)] = 2

#-sinus
# ampl=4
# freq=10
# external_input = ampl/2*(1-np.cos(freq*2*np.pi*t))

#=STIM (Peripherie?)
# stim=np.zeros(tsteps)
stim=double_gaussian(t, 1, 0.01, 0.2, 20)
# stim[int(tsteps*2/4):int(tsteps*3/4)] = 20.
# stim[int(tsteps/2):int(tsteps/2)+50] = 20


#-initial conds
fecont=0;
ficont=0;
we=fecont*b*Tw
wi=ficont*b*Tw
cee,cei,cii=.5,.5,.5
cee1,cii1=0,0


LSwe,LSwi=[],[]
LSfe,LSfi=[],[]
LScee,LScii=[],[]
LScee1,LScii1=[],[]
test,test2=[],[]

for i in progressBar(range(len(t))):
    
    fecontold=fecont
    ficontold=ficont
    # weold,wiold=we,wi
    ceeold,ceiold,ciiold=cee,cei,cii
    TCfe = external_input[i]+stim[i]/8
    REfe = external_input[i]+fecont/16

    # TFs
    Fe = TF('TC',TCfe,ficont,we)
    Fi = TF('RE',REfe,ficont,wi)

    #-TF derivatives
    # df=1e-5
    # dveFe = ( TF('TC',TCfe+df,ficont,we) - Fe )/df
    # dviFe = ( TF('TC',TCfe,ficont+df,we) - Fe )/df
    # dveFi = ( TF('RE',REfe+df,ficont,wi) - Fi )/df
    # dviFi = ( TF('RE',REfe,ficont+df,wi) - Fi )/df
    dveFe = (TF('TC',TCfe+df/2,ficont,we)-TF('TC',TCfe-df/2,ficont,we))/df
    dviFe = (TF('TC',TCfe,ficont+df/2,we)-TF('TC',TCfe,ficont-df/2,we))/df
    dveFi = (TF('RE',REfe+df/2,ficont,wi)-TF('RE',REfe-df/2,ficont,wi))/df
    dviFi = (TF('RE',REfe,ficont+df/2,wi)-TF('RE',REfe,ficont-df/2,wi))/df

    # df=1e-6
    # dvedveFe = ( dveFe*df - Fe + TF('TC',TCfe-df,ficont,we) )/df**2
    # dvidveFe = ( ( TF('TC',TCfe+df,ficont+df,we) - Fe )/df - dveFe )/df
    # dvidviFe = ( dviFe*df - Fe + TF('TC',TCfe,ficont-df,we) )/df**2
    # dvedviFe = ( ( TF('TC',TCfe+df,ficont+df,we) - Fe )/df - dviFe )/df
    # dvedveFi = ( dveFi*df - Fi + TF('RE',REfe-df,ficont,wi) )/df**2
    # dvidveFi = ( ( TF('RE',REfe+df,ficont+df,wi) - Fi )/df - dveFi )/df
    # dvidviFi = ( dviFi*df - Fi + TF('RE',REfe,ficont-df,wi) )/df**2
    # dvedviFi = ( ( TF('RE',REfe+df,ficont+df,wi) - Fi )/df - dviFi )/df
    dvedveFe = ( TF('TC',TCfe+df,ficont,we) - 2*Fe + TF('TC',TCfe-df,ficont,we) )/df**2
    dvidveFe = ( (TF('TC',TCfe+df/2,ficont+df/2,we)-TF('TC',TCfe-df/2,ficont+df/2,we)) - (TF('TC',TCfe+df/2,ficont-df/2,we)-TF('TC',TCfe-df/2,ficont-df/2,we)) )/df**2
    dvidviFe = ( TF('TC',TCfe,ficont+df,we) - 2*Fe + TF('TC',TCfe,ficont-df,we) )/df**2
    dvedviFe = ( (TF('TC',TCfe+df/2,ficont+df/2,we)-TF('TC',TCfe+df/2,ficont-df/2,we)) - (TF('TC',TCfe-df/2,ficont+df/2,we)-TF('TC',TCfe-df/2,ficont-df/2,we)) )/df**2
    dvedveFi = ( TF('RE',REfe+df,ficont,wi) - 2*Fi + TF('RE',REfe-df,ficont,wi) )/df**2
    dvidveFi = ( (TF('RE',REfe+df/2,ficont+df/2,wi)-TF('RE',REfe-df/2,ficont+df/2,wi)) - (TF('RE',REfe+df/2,ficont-df/2,wi)-TF('RE',REfe-df/2,ficont-df/2,wi)) )/df**2
    dvidviFi = ( TF('RE',REfe,ficont+df,wi) - 2*Fi + TF('RE',REfe,ficont-df,wi) )/df**2
    dvedviFi = ( (TF('RE',REfe+df/2,ficont+df/2,wi)-TF('RE',REfe+df/2,ficont-df/2,wi)) - (TF('RE',REfe-df/2,ficont+df/2,wi)-TF('RE',REfe-df/2,ficont-df/2,wi)) )/df**2


    #-first order EULER
    # fecont += dt/T*( (Fe-fecont) )
    # ficont += dt/T*( (Fi-ficont) )

    #-first order HEUN
    # fecont += dt/T*(Fe-fecont)
    # fecont = fecontold + dt/T/2*(Fe-fecontold + TF('TC',TCfe,ficont,we)-fecont)
    # ficont += dt/T*(Fi-ficont)
    # ficont = ficontold + dt/T/2*(Fi-ficontold + TF('RE',REfe,ficont,wi)-ficont)

    #-second order MF
    fecont += dt/T*( (Fe-fecont) + (cee*dvedveFe+cei*dvedviFe+cii*dvidviFe+cei*dvidveFe)/2 )
    ficont += dt/T*( (Fi-ficont) + (cee*dvedveFi+cei*dvedviFi+cii*dvidviFi+cei*dvidveFi)/2 )
    
    #TODO: second order HEUN

    we += dt*MFw('TC',we,fecontold,0) # fecontold = v_e here
    wi += dt*MFw('RE',wi,REfe,ficontold) # ficontold = v_i = finh (for MPF)

    if fecont<1e-9: fecont=1e-9
    if ficont<1e-9: ficont=1e-9
    # if w<=0: w=1e-9

    LSfe.append(float(fecont))
    LSfi.append(float(ficont))
    LSwe.append(float(we))
    LSwi.append(float(wi))


    #-covariances EULER
    cee += dt/T*( Fe*(1/T-Fe)/500 + (Fe-fecontold)**2 + 2*cee*dveFe + 2*ceiold*dviFe - 2*cee)
    cei += dt/T*( (Fe-fecontold)*(Fi-ficontold) + cei*dveFe + ceeold*dveFi + ciiold*dviFe + cei*dviFi - 2*cei)
    cii += dt/T*( Fi*(1/T-Fi)/500 + (Fi-ficontold)**2 + 2*cii*dviFi + 2*ceiold*dveFi - 2*cii)
    # cee1 += dt/T*( Fe*(1/T-Fe)/500 + (Fe-fecontold)**2 -2*cee)#+ 2*cee*dveFe + 2*cei*dviFe - 2*cee)
    # cii1 += dt/T*( Fi*(1/T-Fi)/500 + (Fi-ficontold)**2 -2*cii)#+ 2*cii*dviFi + 2*cei*dveFi - 2*cii)
    #-covs HEUN
    # cee += dt/T*( Fe*(1/T-Fe)/500 + (Fe-fecontold)**2 + 2*cee*dveFe + 2*cei*dviFe - 2*cee)
    # cee = ceeold + dt/T*( Fe*(1/T-Fe)/500 + (Fe-fecontold)**2 + ceeold*dveFe + 2*cei*dviFe - ceeold + cee*dveFe - cee)
    # cei += dt/T*( (Fe-fecontold)*(Fi-ficontold) + cee*dveFi + cei*dveFe + cei*dviFi + cii*dviFe - 2*cei)
    # cei = ceiold + dt/T*( (Fe-fecontold)*(Fi-ficontold) + cee*dveFi + ceiold*dveFe/2 + ceiold*dviFi/2 + cii*dviFe - ceiold + cei*dveFe/2 + cei*dviFi/2 - cei)
    # cii += dt/T*( Fi*(1/T-Fi)/500 + (Fi-ficontold)**2 + 2*cei*dveFi + 2*cii*dviFi - 2*cii)
    # cii = ciiold + dt/T*( Fi*(1/T-Fi)/500 + (Fi-ficontold)**2 + 2*cei*dveFi + ciiold*dviFi - ciiold + cii*dviFi - cii)

    if cee<1e-9: cee=1e-9
    if cii<1e-9: cii=1e-9
    if cei<1e-9: cei=1e-9
    cee=np.sqrt(cee)
    cei=np.sqrt(cei)
    cii=np.sqrt(cii)

    LScee.append(np.sqrt(cee))
    LScii.append(np.sqrt(cii))
    # LScee1.append(np.sqrt(cee1))
    # LScii1.append(np.sqrt(cii1))

    #-test
    # test.append(dvedveFe)
    # test2.append(dvidveFe)


#==================
LSfe=np.array(LSfe)
LSfi=np.array(LSfi)
LSwe=np.array(LSwe)
LSwi=np.array(LSwi)
LScee=np.array(LScee)
LScii=np.array(LScii)

#-SAVE
np.save('data\\MF_out_we', np.vstack((LSfe,LSfi)))
np.save('data\\MF_out_cov', np.vstack((LScee,LScii)))
# np.savetxt('test.txt',test)
# np.save('data\\MF_out_adaptnew', np.vstack((LSfe,LSfi)))


#-testplots
# plt.plot(test)
# plt.plot(test2)
# plt.show()
# plt.plot(LScee, 'b')
# plt.plot(LScee1, '--b')
# plt.plot(LScii, 'r')
# plt.plot(LScii1, '--r')
# plt.show()

#===PLOTS
fig, ax=plt.subplots(2,2, figsize=(10,6), width_ratios=[1,3], height_ratios=[2,1])
fig.delaxes(ax[0,0])

ax[1,0].plot(LSfe, LSfi, c='black')
ax[1,0].plot(LSfe[0], LSfi[0], '.', c='black')
ax[1,0].set_xlabel('LSfe (Hz)')
ax[1,0].set_ylabel('LSfi (Hz)')

ax[0,1].plot(t, LSfe, c='b', label='LSfe')
ax[0,1].fill_between(t, LSfe-LScee, LSfe+LScee, color='b', label='cee', alpha=0.2)
ax[0,1].plot(t, LSfi, c='r', label='LSfi')
ax[0,1].fill_between(t, LSfi-LScii, LSfi+LScii, color='r', label='cii', alpha=0.2)
ax[0,1].plot(t,external_input, c='black', ls='--', label='Dext')
ax[0,1].plot(t,stim, c='black', label='stim')
ax[0,1].legend(loc='upper right')
# ax[0,1].set_xlabel('time (s)')
ax[0,1].set_ylabel('frequencies (Hz)')
ax[0,1].set_ylim(-1,max(np.concatenate([LSfe,LSfi]))+5)

ax[1,1].plot(t, LSwe*1e12, c='b', label='LSwe')
ax[1,1].plot(t, LSwi*1e12, c='r', label='LSwi')
ax[1,1].legend()
ax[1,1].set_xlabel('time (s)')
ax[1,1].set_ylabel('adaptation pA')

plt.tight_layout()
plt.savefig('gfx\\MF_PLOT.png', dpi=250)
