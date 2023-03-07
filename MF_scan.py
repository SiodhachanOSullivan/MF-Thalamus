# Model from  Di Volo et al. Neural Comp. 2019


import numpy as np
import matplotlib.pylab as plt
from scipy.special import erfc
from mytools import progressBar


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
Tw=200*1.e-3
b=0.01*1e-9

Ti=5*1.e-3 # Tsyn RE
Te=5*1.e-3 # Tsyn TC
Ee=0
Ei=-80*1.e-3



T=10e-3



PTC=np.load('DATA\\NEW2params_TC.npy')
PRE=np.load('DATA\\NEW2params_RE.npy')




tfinal=1. # s
dt=5e-4 # s
tsteps=int(tfinal/dt)

t = np.linspace(0, tfinal, tsteps)


scanfe=[]
scanfi=[]

frange=np.linspace(0.01,100,100)

for ff in progressBar(frange):

    fecont=0;
    ficont=0;
    we=fecont*b*Tw
    wi=ficont*b*Tw
    
    external_input=np.full(tsteps, ff)
    # external_input+=np.abs(np.random.randn(tsteps))*5

    LSwe=[]
    LSwi=[]
    LSfe=[]
    LSfi=[]
    for i in range(len(t)):
        
        fecontold=fecont
        ficontold=ficont
        # weold,wiold=we,wi
        TCfe = external_input[i]#+stim[i]/8
        REfe = external_input[i]+fecont/16

        # TFs
        Fe = TF('TC',TCfe,ficont,we)
        Fi = TF('RE',REfe,ficont,wi)

        #-TF derivatives
        # df=1e-5
        # dveFe = ( TF('TC',TCfe+df/24,ficont,we) - Fe )/df/24
        # dviFe = ( TF('TC',TCfe,ficont+df,we) - Fe )/df
        # dveFi = ( TF('RE',REfe+df/12,ficont,wi) - Fi )/df/12
        # dviFi = ( TF('RE',REfe,ficont+df/6,wi) - Fi )/df/6

        #-first order EULER
        fecont += dt/T*( (Fe-fecont) )
        ficont += dt/T*( (Fi-ficont) )

        we += dt*MFw('TC',we,fecontold,0) # fecontold = v_e here
        wi += dt*MFw('RE',wi,REfe,ficontold) # ficontold = v_i = finh (for MPF)

        if fecont<1e-9: fecont=1e-9
        if ficont<1e-9: ficont=1e-9

        LSfe.append(float(fecont))
        LSfi.append(float(ficont))

    scanfe.append(np.mean(LSfe))
    scanfi.append(np.mean(LSfi))



#-SAVE
np.save('data\\MF_out_scan', np.vstack((scanfe,scanfi)))

#-PLOT
plt.plot(frange, scanfe, 'g')
plt.plot(frange, scanfi, 'r')
# plt.plot(frange, frange, c='black', ls='--')
plt.show()
