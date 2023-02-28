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



PTC=np.load('DATA\\NEW2params_TC.npy')
PRE=np.load('DATA\\NEW2params_RE.npy')




tfinal=1. # s
dt=1e-3 # s
tsteps=int(tfinal/dt)

t = np.linspace(0, tfinal, tsteps)



fecont=0;
ficont=0;
w=ficont*b*Tw


scanfe=[]
scanfi=[]

frange=np.linspace(0.01,2,100)

for ff in progressBar(frange):
    
    external_input=np.full(tsteps, ff)
    # external_input+=np.abs(np.random.randn(tsteps))*5

    LSw=[]
    LSfe=[]
    LSfi=[]
    for i in range(len(t)):
        
        fecontold=fecont
        ficontold=ficont

        Fe = TF(PTC,external_input[i],ficont,0, Nexc=800,Ninh=25,Qe=1e-9,Qi=6e-9,Cm=160e-12,El=-65e-3)
        Fi = TF(PRE,external_input[i]+fecont/16,ficont,w, Nexc=400,Ninh=150,Qe=4e-9,Qi=1e-9,Cm=200e-12,El=-75e-3)

        fecont += dt/T*(Fe-fecont)
        ficont += dt/T*(Fi-ficont)
        w += dt*(-w/Tw+b*ficontold)

        LSfe.append(float(fecont))
        LSfi.append(float(ficont))
        LSw.append(float(w))

    scanfe.append(np.mean(LSfe))
    scanfi.append(np.mean(LSfi))



#-SAVE
np.save('data\\MF_out_scan', np.vstack((scanfe,scanfi)))

#-PLOT
plt.plot(frange, scanfe, 'g')
plt.plot(frange, scanfi, 'r')
# plt.plot(frange, frange, c='black', ls='--')
plt.show()
