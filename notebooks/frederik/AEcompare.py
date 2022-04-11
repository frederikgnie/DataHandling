#%%
import os
from DataHandling import plots, utility
from DataHandling.features import slices
import shutil
import numpy as np
from tensorflow import keras

domain = 'blonigan'

# Load model prediction/target
if domain == 'nakamura':
    name=["cosmic-feather-29","valiant-river-31","deep-leaf-32"]
elif domain == 'blonigan':
    name=["swift-sky-34","volcanic-gorge-35","generous-flower-36"]

predlist=[]
for i in name:
    model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(i))
    pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(i))
    predlist.append(pred["test"])
targ = np.load('/home/au569913/DataHandling/models/output/{}/targets.npz'.format(name[0]))   

target=targ["test"] 
# %% Plot rms
import xarray as xr
from DataHandling import plots
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
plots.rmsplot('AE',target,predlist[0],predlist[1],predlist[2],ds,domain)

# %%
from DataHandling import postprocess
import importlib
importlib.reload(postprocess)
AE_error=[]
for i in range(0,3):
    AE_error.append(postprocess.errornorm(predlist[i],target))
POD_error = [postprocess.errornorm(c1,test_snap), postprocess.errornorm(c2,test_snap),postprocess.errornorm(c3,test_snap)]

# %%
