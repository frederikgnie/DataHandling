#%%
import os
from DataHandling import plots, utility
from DataHandling.features import slices
import shutil
import numpy as np
from tensorflow import keras
from DataHandling import POD
from DataHandling import postprocess
from DataHandling.features import preprocess
import importlib
#Define domain
domain = 'blonigan'

import xarray as xr
from DataHandling import plots
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
#%%
# Load model prediction/target - # from least to most compressed
if domain == 'nakamura':
    name=["cosmic-feather-29","valiant-river-31","deep-leaf-32"]
elif domain == 'blonigan':
    name=["swift-sky-34","volcanic-gorge-35","generous-flower-36"]
    scae=["fragrant-flower-71",'dandy-bush-79','northern-capybara-82']
elif domain == '1pi':
    name=["ethereal-snow-37","confused-waterfall-38","noble-wind-39"]

predlist=[]
predscae=[]
for i,j in zip(name,scae):
    model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(i))
    pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(i))
    predlist.append(pred["test"])

    model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(j))
    pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(j))
    predscae.append(pred["test"])
    
targ = np.load('/home/au569913/DataHandling/models/output/{}/targets.npz'.format(name[0]))   

target=targ["test"] 

# %% import POD
modes = [1536, 192, 24]
if domain == '1pi':
    modes = [3072,384,48]

c1,d1 = POD.projectPOD(modes[0],domain)
c2,d2 = POD.projectPOD(modes[1],domain)
c3,d3 = POD.projectPOD(modes[2],domain)
c = [c1,c2,c3]
d = [d1,d2,d3]
test_snap = np.load("/home/au569913/DataHandling/models/POD/{}/test_snap.npy".format(domain))
# %% Plot rms

plots.rmsplot('POD',test_snap,c1,c2,c3,ds,domain)
plots.rmsplot('CNNAE',target,predlist[0],predlist[1],predlist[2],ds,domain)
plots.rmsplot('SCAE',target,predscae[0],predscae[1],predscae[2],ds,domain,scae=scae)

# %% L2 error calculations and plot
importlib.reload(postprocess)
importlib.reload(plots)
CNNAE_error=[]
SCAE_error=[]
for i in range(0,3):
    CNNAE_error.append(postprocess.errornorm(predlist[i],target))
    SCAE_error.append(postprocess.errornorm(predscae[i],target))
POD_error = [postprocess.errornorm(c1,test_snap), postprocess.errornorm(c2,test_snap),postprocess.errornorm(c3,test_snap)]
plots.errorplot(POD_error,CNNAE_error,SCAE_error,domain,scae,save=True)

#%% Arranged TKE plot
importlib.reload(plots)
test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
u_tau=0.05
TKE_total=postprocess.KE_ds(preprocess.flucds(ds.isel(time=test_ind)))/(u_tau**2) #calculate 
TKE_pred_total=postprocess.KE_np(predlist[2],ds)
TKE_POD = postprocess.KE_np(c[2],ds)
TKE_SCAE = postprocess.KE_np(predscae[2],ds)
#Plot
plots.KE_arrangeplot(TKE_total, TKE_pred_total, TKE_POD,TKE_SCAE, domain,showscae=False,save=True)

abserror_high,relerror_high = plots.KE_arrange5(TKE_total, TKE_pred_total, TKE_POD, domain, TKE_SCAE, cut='high', showscae=True, save=False)
abserror_low,relerror_low  = plots.KE_arrange5(TKE_total, TKE_pred_total, TKE_POD, domain, TKE_SCAE, cut='low', showscae=True, save=False)

# %% Target prediction isocontours, here sorted by TKE
TKE_min = TKE_total.argmin().values #index 
TKE_max = TKE_total.argmax().values #index

test_ind_toplot = TKE_max
data = postprocess.Qcrit(target*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'Target',domain,'Qcrit',save=False)
data = postprocess.Qcrit(predlist[2]*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'CNNAE prediction',domain,'Qcrit',save=False)
# %% Uslice 4
importlib.reload(plots)
u_tau=0.05
plots.uslice4(target,c3/u_tau,predlist[2],predscae[0],'hard',domain,'z',save=False)
plots.uslice4(target,c3/u_tau,predlist[2],predscae[0],'hard',domain,'y',save=False)

# %%
