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
from keras.utils.layer_utils import count_params
#Define domain
domain = '1pi'

import xarray as xr
from DataHandling import plots
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
#%%
# Load model prediction/target - # from least to most compressed
if domain == 'nakamura':
    name=["cosmic-feather-29","valiant-river-31","deep-leaf-32"]
    scae = ["smooth-microwave-115","radiant-oath-96","desert-breeze-97","misty-night-98"] #lambda,e-2, e-1, 1, e1
elif domain == 'blonigan':
    name=["swift-sky-34","volcanic-gorge-35","generous-flower-36"]
    scae=["dulcet-armadillo-70","fragrant-flower-71",'dandy-bush-79','northern-capybara-82'] #here 4 
elif domain == '1pi':
    name=["ethereal-snow-37","confused-waterfall-38","noble-wind-39"]
    scae=["sage-blaze-101","curious-blaze-103","polar-paper-104"] #lambda e-2, e-1, 1

predlist=[]
predscae=[]
scaeparams=[]
cnnaeparams=[]
for i,j in zip(name,scae):
    model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(i))
    cnnaeparams.append(count_params(model.trainable_weights))
    pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(i))
    predlist.append(pred["test"])

    model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(j))
    scaeparams.append(count_params(model.trainable_weights))
    pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(j))
    predscae.append(pred["test"])
    
targ = np.load('/home/au569913/DataHandling/models/output/{}/targets.npz'.format(name[1]))   

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
print(POD_error)
print(CNNAE_error)
print(SCAE_error)
#%% Arranged TKE plot
importlib.reload(plots)
test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
u_tau=0.05

# Only for test & for reduced domain.  
TKE_total=postprocess.KE_ds(preprocess.flucds(ds.isel(time=test_ind)))/(u_tau**2) #calculate 
TKE_pred_total=postprocess.KE_np(predlist[2],ds)
TKE_POD = postprocess.KE_np(c[2],ds)
TKE_SCAE = postprocess.KE_np(predscae[2],ds)

# Plot
plots.KE_arrangeplot(TKE_total, TKE_pred_total, TKE_POD,TKE_SCAE, domain,showscae=True,save=True)

abserror_high,relerror_high = plots.KE_arrange5(TKE_total, TKE_pred_total, TKE_POD, domain, TKE_SCAE, cut='high', showscae=True, save=True)
abserror_low,relerror_low  = plots.KE_arrange5(TKE_total, TKE_pred_total, TKE_POD, domain, TKE_SCAE, cut='low', showscae=True, save=True)

# %% Target prediction isocontours, here sorted by TKE
#TKE_min = TKE_total.argmin().values #index 
#TKE_max = TKE_total.argmax().values #index

#test_ind_toplot = TKE_min
test_ind_toplot = 1
data = postprocess.Qcrit(target*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'Target',domain,'Qcrit',save=True)
data = postprocess.Qcrit(c[2],ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'POD prediction',domain,'Qcrit',save=True)
data = postprocess.Qcrit(predlist[2]*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'CNNAE prediction',domain,'Qcrit',save=True)
data = postprocess.Qcrit(predscae[2]*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'SCAE prediction',domain,'Qcrit',save=True)


# %% Uslice 4
importlib.reload(plots)
u_tau=0.05
plots.uslice4(target,c[2]/u_tau,predlist[2],predscae[2],'hard',domain,'z',save=True)
plots.uslice4(target,c[2]/u_tau,predlist[2],predscae[2],'hard',domain,'y',save=True)

# %% Dissipation
# %%
#from DataHandling import postprocess
#import importlib
#importlib.reload(postprocess)
#diss_CNNAE = postprocess.diss(predlist[0], ds)
#diss_tar = postprocess.diss(target, ds)
#diss_POD = postprocess.diss(c[0]/u_tau, ds)
#diss_SCAE = postprocess.diss(predscae[0], ds)
#importlib.reload(plots)
#u_tau = 0.05
#arr1inds = diss_tar.argsort()
#plots.diss_arrangeplot(diss_tar, diss_CNNAE, diss_POD, diss_SCAE, domain,showscae=True,save=False)
#plots.diss_arrangeerror(diss_tar, diss_CNNAE, diss_POD, diss_SCAE, domain,showscae=True,save=False)


# %%
importlib.reload(plots)
TKE_total=postprocess.KE_np(target,ds) #calculate 
TKE_pred_total=postprocess.KE_np(predlist[2],ds)
TKE_POD = postprocess.KE_np(c[2],ds)
TKE_SCAE = postprocess.KE_np(predscae[2],ds)

plots.TKE_arrangeerror(TKE_total, TKE_pred_total, TKE_POD, TKE_SCAE, domain, showscae=True,save=True)
# %%
u_tau = 0.05
POD_err = abs(TKE_total-TKE_POD/(u_tau**2))
POD_arrange = POD_err.argsort()
test_ind = np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

time_to_plot = ds.isel(time=test_ind).coords['time'][POD_arrange[-5:]].values
data = postprocess.Qcrit('ds',ds, time_to_plot[4]) # Calc q-criterion
plots.isocon(data,ds,time_to_plot[4],domain,'Qcrit',save=True)

time_to_plot = ds.isel(time=test_ind).coords['time'][POD_arrange[245:250]].values
data = postprocess.Qcrit('ds',ds, time_to_plot[3]) # Calc q-criterion
plots.isocon(data,ds,time_to_plot[3],domain,'Qcrit',save=True)

time_to_plot = ds.isel(time=test_ind).coords['time'][POD_arrange[0:5]].values
data = postprocess.Qcrit('ds',ds, time_to_plot[0]) # Calc q-criterion
plots.isocon(data,ds,time_to_plot[0],domain,'Qcrit',save=True)

# %%
importlib.reload(plots)
KE_total= xr.open_dataarray("KE_{}.nc".format(domain))
ind_to_plot = [test_ind[POD_arrange[-5:]],test_ind[POD_arrange[245:250]],test_ind[POD_arrange[0:5]]]
plots.KE_plot(KE_total,domain,fluc=False,KE_pred=ind_to_plot,vlines=True,save=True)
# %%


# %%
