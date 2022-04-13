#%%
import os
from DataHandling import plots, utility
from DataHandling.features import slices
import shutil
import numpy as np
from tensorflow import keras
var=['u_vel','v_vel','w_vel']
target=['u_tar','v_tar','w_tar']
y_plus=15
normalized=False

# Load model prediction/target
name="mild-pine-42" #l1=1e-5
model_type = 'SCAE'
model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(name))

#model.summary()
#pred = np.load('/home/au569913/DataHandling/models/output/{}/y_plus_15-VARS-u_vel_v_vel_w_vel-TARGETS-u_tar_v_tar_w_tar/predictions.npz'.format(name))
#targ = np.load('/home/au569913/DataHandling/models/output/{}/y_plus_15-VARS-u_vel_v_vel_w_vel-TARGETS-u_tar_v_tar_w_tar/targets.npz'.format(name))
pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(name))
targ = np.load('/home/au569913/DataHandling/models/output/{}/targets.npz'.format(name))

target_list=[targ["train"],targ["val"],targ["test"]]
predctions=[pred["train"],pred["val"],pred["test"]]
#%% GENERAL PLOTS FOR THE DOMAIN (NO-MODEL) %%%%%%%%%
import xarray as xr
domain = 'blonigan'
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
#ds=ds.isel(y=slice(0, 32))

# %% Plot domain vel contour
from DataHandling import plots
import importlib
importlib.reload(plots)
plots.dsfield(ds,domain,dim='z')
plots.dsfield(ds,domain,dim='y')
#%% Kinetic energy plot of full domain
import xarray as xr
from DataHandling.features import preprocess
from DataHandling import postprocess
from DataHandling import plots
import numpy as np
import matplotlib.pyplot as plt
u_tau = 0.05
#ds = preprocess.flucds(ds)/u_tau   #utilise fluctuations or not

KE_total=postprocess.KE_ds(ds) #calculate KE for all timesteps
KE_total = KE_total/(u_tau**2) #nondimensionalize
KE_min = KE_total.isel(time=KE_total.argmin()).coords['time'].values #index 
KE_max = KE_total.isel(time=KE_total.argmax()).coords['time'].values #index
#Predictions to be scattered if needed
#KE_pred_total=postprocess.KE_np(predctions[2],ds) #pick out train/val/test
plots.KE_plot(KE_total,domain,fluc=False,KE_pred=False)
# %% Isocountours  #######
from DataHandling import postprocess
from DataHandling import plots
import importlib
importlib.reload(postprocess)
importlib.reload(plots)
#data = ds['u_vel'][200].values #pick vel field
#plots.isocon(data,ds,'u_vel=200','nakamura','Qcrit')

#%% Min/max kinetic energy isocontours of q crit
importlib.reload(postprocess)
importlib.reload(plots)
#Nakamura
#KE_max = 3003.
#KE_min = 16182 
#Blonigan
KE_max = 4269
KE_min = 13056

data = postprocess.Qcrit('ds',ds,KE_max) # Calc q-criterion
plots.isocon(data,ds,KE_max,domain,'Qcrit')
data = postprocess.Qcrit('ds',ds,KE_min) # Calc q-criterion
plots.isocon(data,ds,KE_min,domain,'Qcrit')

#%%  MODEL SPECIFIC %%%%%%%%%
ds=ds.isel(y=slice(0, 32))

#%% Prediction target plots uslice

#velocity
import DataHandling
from DataHandling import plots
import importlib
importlib.reload(plots)
plots.uslice(predctions[2],target_list[2],'CNNAE',domain,'z')
plots.uslice(predctions[2],target_list[2],'CNNAE',domain,'y')

#%% Isocon For target/pred of test_ind
from DataHandling import postprocess
importlib.reload(plots)
u_tau=0.05
test_ind_toplot = 1 #30   #test_ind[208]=ind number 305 in time = time 3918 
data = postprocess.Qcrit(target_list[2]*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'Target',domain,'Qcrit')
data = postprocess.Qcrit(predctions[2]*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'AE prediction',domain,'Qcrit')

#%% Arranged TKE plot
from DataHandling import POD
from DataHandling import postprocess
from DataHandling.features import preprocess
test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
u_tau=0.05
TKE_total=postprocess.KE_ds(preprocess.flucds(ds.isel(time=test_ind)))/(u_tau**2) #calculate 
TKE_pred_total=postprocess.KE_np(predctions[2],ds)
modes = 24
c3,d3 = POD.projectPOD(modes,domain)
TKE_c3 = postprocess.KE_np(c3,ds)
plots.KE_arrangeplot(TKE_total, TKE_pred_total, TKE_c3,domain)


