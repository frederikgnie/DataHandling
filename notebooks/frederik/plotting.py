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
name = 'fearless-shadow-54' #l1=1e-3
name = 'fragrant-flower-71' #l1=1
name = 'northern-capybara-82' #l1=1e2

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

KE_total=postprocess.KE_ds(ds) #calculate KE for all timesteps
KE_total = KE_total/(u_tau**2) #nondimensionalize
#Pick time of highest and lowest kinetic energy
KE_min = KE_total.isel(time=KE_total.argmin()).coords['time'].values #time
KE_max = KE_total.isel(time=KE_total.argmax()).coords['time'].values #time
#Pick index of snapshot with highest and lowest energy of test values
test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
KE_min_test = KE_total.isel(time=test_ind).argmin().values #index 
KE_max_test = KE_total.isel(time=test_ind).argmax().values #index 

#Predictions to be scattered if needed
#KE_pred_total=postprocess.KE_np(predctions[2],ds) #pick out train/val/test
plots.KE_plot(KE_total,domain,fluc=False,KE_pred=False)
#TKE
TKE_total=postprocess.KE_ds(preprocess.flucds(ds))/(u_tau**2) #calculate 
plots.KE_plot(TKE_total,domain,fluc=True,KE_pred=False)
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
#KE_min = 5367 
#Blonigan
#KE_max = 4269
#KE_min = 13056

data = postprocess.Qcrit('ds',ds,KE_max) # Calc q-criterion
plots.isocon(data,ds,KE_max,domain,'Qcrit')
data = postprocess.Qcrit('ds',ds,KE_min) # Calc q-criterion
plots.isocon(data,ds,KE_min,domain,'Qcrit')
#Isocontour for time of least Q-crit in flow (blonigan)
#t_least = 3846
#data = postprocess.Qcrit('ds',ds,t_least)
#plots.isocon(data,ds,t_least,domain,'Qcrit')
#%% Find times where Q-crit is highest
Qlist = []
for i in np.linspace(3003,6000,1000):
    Qlist.append(np.sum(postprocess.Qcrit('ds',ds,i)))


#%%
#Qmax/min_time 
# #nakamura: 5703,3636  #blon: 3846,4401 #1pi: 
# a = "neg" if b<0 else "pos" if b>0 else "zero"
Qmax = max(Qlist)
Qmax_ind = np.argmax(Qlist)
Qmax_time = np.linspace(3003,6000,1000)[Qmax_ind]
Qmin_ind = min(Qlist)
Qmin_ind = np.argmin(Qlist)
Qmin_time = np.linspace(3003,6000,1000)[Qmin_ind]

#%%  MODEL SPECIFIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ds=ds.isel(y=slice(0, 32))

#%% Prediction target plots uslice
#velocity
import DataHandling
from DataHandling import plots
import importlib
importlib.reload(plots)
plots.uslice(predctions[2],target_list[2],model_type,domain,'z')
plots.uslice(predctions[2],target_list[2],model_type,domain,'y')

#%% Isocon For target/pred of test_ind
from DataHandling import postprocess
importlib.reload(plots)
u_tau=0.05

test_ind_toplot = 1 #30   #test_ind[208]=ind number 30 in time = time 3918 
data = postprocess.Qcrit(target_list[2]*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'Target',domain,'Qcrit',save=True)
data = postprocess.Qcrit(predctions[2]*u_tau,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'SCAE prediction',domain,'Qcrit')



#%%
