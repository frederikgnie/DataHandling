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
#%%
#%%
#name="worthy-glade-62"
#overwrite=False
#var=['u_vel','v_vel','w_vel']
#target=['u_vel','v_vel','w_vel']
#normalized=False
#y_plus=15
model=keras.models.load_model("/home/au569913/DataHandling/models/trained/copper-leaf-23")

pred = np.load('/home/au569913/DataHandling/models/output/copper-leaf-23/y_plus_15-VARS-u_vel_v_vel_w_vel-TARGETS-u_tar_v_tar_w_tar/predictions.npz')
targ = np.load('/home/au569913/DataHandling/models/output/copper-leaf-23/y_plus_15-VARS-u_vel_v_vel_w_vel-TARGETS-u_tar_v_tar_w_tar/targets.npz')
target_list=[targ["train"],targ["val"],targ["test"]]
predctions=[pred["train"],pred["val"],pred["test"]]
#%%
print(np.shape(predctions[1]))
print(np.shape(target_list[0]))

#%%
import DataHandling
from DataHandling import plots
import importlib
importlib.reload(plots)
plots.uslice(predctions,target_list,'wheretosave',ds,'z')

# %% plot figure
import matplotlib.pyplot as plt
uin = ds.u_vel.isel(time=200,z=16)
fig = plt.figure()
ax = fig.add_subplot(111)
#air2d.T.plot(cmap='jet',vmin=0)
uin.T.plot.contourf(ax=ax,levels=200,cmap='jet',vmin=0)
ax.set_aspect('equal')

# %%
import xarray as xr
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/data.zarr")
ds=ds.isel(y=slice(0, 32))


# %%
from DataHandling import postprocess
from DataHandling import plots
import importlib
importlib.reload(postprocess)
importlib.reload(plots)

#data = ds['u_vel'][200].values #pick vel field
#data = postprocess.Qcrit(target_list[2],ds,100) # Calc q-criterion
#plots.isocon(data,ds,'Target')
#data = postprocess.Qcrit(predctions[2],ds,100) # Calc q-criterion
plots.isocon(data,ds,'Prediction')

# %%
