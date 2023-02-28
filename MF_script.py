# Model from  Di Volo et al. Neural Comp. 2019


import numpy as np
import matplotlib.pylab as plt
from scipy.special import erfc
from mytools import progressBar


def TF(P,fexc,finh,adapt, Nexc,Ninh,Qe,Qi,Cm,El):

    fe = fexc*Nexc
    fi = finh*Ninh

    fe+=1e-9;
    fi+=1e-9;
    
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
    
    
    fe+=1e-9;
    fi+=1e-9;

    Tv = ( fe*(Ue*Te)*(Ue*Te) + fi*(Qi*Ui)*(Qi*Ui)) /( fe*(Ue*Te)*(Ue*Te)/(Te+Tm) + fi*(Qi*Ui)*(Qi*Ui)/(Ti+Tm) );
    TvN = Tv*Gl/Cm;
    
    muV0=-60e-3;
    DmuV0 = 10e-3;
    sV0 =4e-3;
    DsV0= 6e-3;
    TvN0=0.5;
    DTvN0 = 1.;

    vthre = P[0] + P[1]*(muV-muV0)/DmuV0 + P[2]*(sV-sV0)/DsV0 + P[3]*(TvN-TvN0)/DTvN0 \
        + P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0) + P[5]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0 + P[6]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0 + P[7]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0) + P[8]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0 + P[9]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0);


    frout = 1/(2*Tv) * erfc( (vthre - muV)/(np.sqrt(2)*sV) )
    
    return frout;





# constants
Gl=10*1.e-9
Tw=200*1.e-3
b=0.01*1e-9

Ti=5*1.e-3 # Tsyn RE
Te=5*1.e-3 # Tsyn TC
Ee=0
Ei=-80*1.e-3



T=0.01



PTC=np.load('data\\NEW2params_TC.npy')
PRE=np.load('data\\NEW2params_RE.npy')




tfinal=1. # s
dt=1e-4 # s
tsteps=int(tfinal/dt)

t = np.linspace(0, tfinal, tsteps)

#=== CORTEX external input
# constant
external_input=np.full(tsteps, 4.)
# external_input+=np.random.randn(tsteps)/2

# timeframe
# external_input=np.zeros(tsteps)
# external_input[:1000] = 4
# external_input[int(tsteps*2/4):int(tsteps*3/4)] = 2

# sinus
# ampl=4
# freq=10
# external_input = ampl/2*(1-np.cos(freq*2*np.pi*t))


#=== STIM (Peripherie?)
stim=np.zeros(tsteps)
# stim[int(tsteps*2/4):int(tsteps*3/4)] = 20.
# stim[int(tsteps/2):int(tsteps/2)+50] = 20



fecont=0;
ficont=0;
w=ficont*b*Tw
cee,cei,cii,cie=0,0,0,0


LSw=[]
LSfe,LSfi=[],[]
LScee,LScii=[],[]
dvidviFe,dvedveFi,dvedviFi,dvidviFi,dvidveFi=0,0,0,0,0

for i in progressBar(range(len(t))):
    
    fecontold=fecont
    ficontold=ficont
    wold = w

    Fe = TF(PTC,external_input[i]+stim[i]/8,ficont,0, Nexc=800,Ninh=25,Qe=1e-9,Qi=6e-9,Cm=160e-12,El=-65e-3)
    Fi = TF(PRE,external_input[i]+fecont/16,ficont,w, Nexc=400,Ninh=150,Qe=4e-9,Qi=1e-9,Cm=200e-12,El=-75e-3)

    # first order MF
    # fecont += dt/T*(Fe-fecont)
    # ficont += dt/T*(Fi-ficont)
    # w += dt*(-w/Tw+b*ficontold)

    # second order MF
    fecont += dt/T*( (Fe-fecont) + cii*dvidviFe/2 )
    # ficont += dt/T*( (Fi-ficont) + (cee*dvedveFi+cei*dvedviFi+cii*dvidviFi+cie*dvidveFi)/2 )
    ficont += dt/T*( (Fi-ficont) + (cee*dvedveFi+cii*dvidviFi)/2 )
    # w += dt*(-w/Tw+b*ficontold)

    LSfe.append(float(fecont))
    LSfi.append(float(ficont))
    LSw.append(float(w))

    # derivatives
    dviFe = ( TF(PTC,external_input[i]+stim[i]/8,ficont,0, Nexc=800,Ninh=25,Qe=1e-9,Qi=6e-9,Cm=160e-12,El=-65e-3) - Fe )/(ficont-ficontold)
    dveFi = ( TF(PRE,external_input[i]+fecont/16,ficontold,wold, Nexc=400,Ninh=150,Qe=4e-9,Qi=1e-9,Cm=200e-12,El=-75e-3) - Fi )/(fecont-fecontold)
    dviFi = ( TF(PRE,external_input[i]+fecontold/16,ficont,wold, Nexc=400,Ninh=150,Qe=4e-9,Qi=1e-9,Cm=200e-12,El=-75e-3) - Fi )/(ficont-ficontold)
    dvidviFe = ( dviFe*(ficont-ficontold) - Fe + TF(PTC,external_input[i]+stim[i]/8,2*ficontold-ficont,0, Nexc=800,Ninh=25,Qe=1e-9,Qi=6e-9,Cm=160e-12,El=-65e-3) )/(ficont-ficontold)**2
    dvedveFi = ( dveFi*(fecont-fecontold) - Fi + TF(PRE,external_input[i]+(2*fecontold-fecont)/16,ficontold,wold, Nexc=400,Ninh=150,Qe=4e-9,Qi=1e-9,Cm=200e-12,El=-75e-3) )/(fecont-fecontold)**2
    # dvidveFi = ( ( TF(PRE,external_input[i]+fecont/16,ficont,wold, Nexc=400,Ninh=150,Qe=4e-9,Qi=1e-9,Cm=200e-12,El=-75e-3) - Fi )/(fecont-fecontold) - dveFi )/(ficont-ficontold)
    dvidviFi = ( dviFi*(ficont-ficontold) - Fi + TF(PRE,external_input[i]+fecontold/16,2*ficontold-ficont,wold, Nexc=400,Ninh=150,Qe=4e-9,Qi=1e-9,Cm=200e-12,El=-75e-3) )/(ficont-ficontold)**2
    # dvedviFi = ( ( TF(PRE,external_input[i]+fecont/16,ficont,wold, Nexc=400,Ninh=150,Qe=4e-9,Qi=1e-9,Cm=200e-12,El=-75e-3) - Fe )/(ficont-ficontold) - dviFi )/(fecont-fecontold)
    # covariances
    # cee += dt/T*( Fe*(1/T-Fe)/500 + (Fe-fecontold)**2 + cei*dveFi + cie*dveFi - 2*cee)
    # cei += dt/T*( (Fe-fecontold)*(Fi-ficontold) + cei*dveFi + cei*dviFe + cii*dviFi- 2*cei)
    # cii += dt/T*( Fi*(1/T-Fi)/500 + (Fi-ficontold)**2 + cie*dviFe + 2*(cii*dviFi) + cei*dviFe - 2*cii)
    # cie += dt/T*( (Fi-ficontold)*(Fe-fecontold) + cie*dviFe + cii*dviFi + cie*dveFi - 2*cie)
    # covariances SIMPLE
    cee += dt/T*( Fe*(1/T-Fe)/500 + (Fe-fecontold) - 2*cee)
    cii += dt/T*( Fi*(1/T-Fi)/500 + (Fi-ficontold) + (cii*dviFi)**2 - 2*cii)

    LScee.append(float(cee))
    LScii.append(float(cii))

LSfe=np.array(LSfe)
LSfi=np.array(LSfi)
LScee=np.array(LScee)
LScii=np.array(LScii)
#-SAVE
# np.save('data\\MF_out_stim', np.vstack((LSfe,LSfi)))
# np.save('data\\MF_out_cov', np.vstack((LScee,LScii)))


# plt.plot(LScee, c='black')
# plt.show()

fig=plt.figure(figsize=(12,4))
ax = fig.add_subplot(1, 2, 1, projection = '3d')
ax.plot(LSfe, LSfi, np.array(LSw)*1e12, c='black')
ax.set_xlabel('LSfe (Hz)')
ax.set_ylabel('LSfi (Hz)')
ax.set_zlabel('LSw (pA)')
# plt.savefig('Phasespace3D.png')

ax = fig.add_subplot(1,2,2)
ax.plot(t, LSfe, c='b', label='fe')
ax.fill_between(t, LSfe-LScee, LSfe+LScee, color='b', label='cee', alpha=0.2)
ax.plot(t, LSfi, c='r', label='fi')
ax.fill_between(t, LSfi-LScii, LSfi+LScii, color='r', label='cii', alpha=0.2)
ax.plot(t,external_input, c='black', ls='--', label='extI')
ax.legend()
ax.set_xlabel('time (s)')
ax.set_ylabel('frequencies (Hz)')

plt.tight_layout()
plt.savefig('gfx\\MF_PLOT.png')

