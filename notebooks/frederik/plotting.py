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

#%% Load model prediction/target
name="cosmic-feather-29"

model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(name))
#model.summary()
#pred = np.load('/home/au569913/DataHandling/models/output/{}/y_plus_15-VARS-u_vel_v_vel_w_vel-TARGETS-u_tar_v_tar_w_tar/predictions.npz'.format(name))
#targ = np.load('/home/au569913/DataHandling/models/output/{}/y_plus_15-VARS-u_vel_v_vel_w_vel-TARGETS-u_tar_v_tar_w_tar/targets.npz'.format(name))
pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(name))
targ = np.load('/home/au569913/DataHandling/models/output/{}/targets.npz'.format(name))


target_list=[targ["train"],targ["val"],targ["test"]]
predctions=[pred["train"],pred["val"],pred["test"]]


#%% velocity slice
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

# %% Isocountours
from DataHandling import postprocess
from DataHandling import plots
import importlib
importlib.reload(postprocess)
importlib.reload(plots)

#data = ds['u_vel'][200].values #pick vel field
data = postprocess.Qcrit(target_list[2],ds,50) # Calc q-criterion
plots.isocon(data,ds,'Target')
data = postprocess.Qcrit(predctions[2],ds,50) # Calc q-criterion
plots.isocon(data,ds,'Prediction')

# %% vel_rms plots
from DataHandling.features import preprocess
import matplotlib.pyplot as plt
import importlib
importlib.reload(preprocess)
#fluc = preprocess.flucnp(target_list[2])
#rms_tar = preprocess.rms(fluc)
#fluc = preprocess.flucnp(predctions[2])
#rms_pred = preprocess.rms(fluc)
rms_tar = preprocess.rms(target_list[2])
rms_pred = preprocess.rms(predctions[2])
u_tau = 0.05
y = ds.coords['y'].values
#plt.plot(y,rms_tar[:,2]/u_tau)
#plt.plot(y,rms_pred[:,2]/u_tau)
plt.plot(y,rms_tar[:,1])
plt.plot(y,rms_pred[:,1])

#%% Kinetic energy
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/data.zarr")
ds=ds.isel(y=slice(0, 32))
u_tau = 0.05
ds = preprocess.flucds(ds)/u_tau
# pick out coords array
x = ds.coords['x'].values
y = ds.coords['y'].values
z = ds.coords['z'].values
#ds = ds.reindex(y=ds.y[::-1])
ds = ds.reindex(y=list(reversed(ds.y))) #to not get negative integration values for y
# Map all inner product 
ds = ds.assign(KE=lambda ds: 0.5*(ds['u_vel']*ds['u_vel']+ds['v_vel']*ds['v_vel']+ds['w_vel']*ds['w_vel']))
ds = ds['KE'] #pick out only KE data array
ds =ds.chunk('auto')
ds = ds.integrate('y')
ds = ds.integrate('x')
ds = ds.integrate('z')
print('Loading')
ds = ds.load() #convert to np array
KE_total = ds

# %%
data = predctions[2] #pick out train/val/test
KE_pred = np.zeros(shape=(np.shape(data[:,0,0,0,0])[0],32,32,32)) #initialise

for l in range(np.shape(KE_pred[:,0,0,0])[0]):
    for i in range(32):
        for j in range(32):
            for k in range(32):
                KE_pred[l,i,j,k] = 0.5*(data[l,i,j,k,0]*data[l,i,j,k,0]+data[l,i,j,k,1]*data[l,i,j,k,1]+data[l,i,j,k,2]*data[l,i,j,k,2])
#%%
#integrate
KE_pred_total = np.trapz(KE_pred,x,axis=1)
KE_pred_total = np.trapz(KE_pred_total,z,axis=2)
KE_pred_total = -np.trapz(KE_pred_total,y,axis=1)

test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
test_time = ds.coords['time'][test_ind]

train_ind =np.load("/home/au569913/DataHandling/data/interim/train_ind.npy")
train_time = ds.coords['time'][train_ind]

#KE_total.plot(color='k',lw=0.5)
plt.scatter(test_time,KE_pred_total*(u_tau**2),marker='.')

#%%
#KE_pred_total_sort = KE_pred_total.sort() #sorted array of predictions
#KE_total_sort=KE_total.sortby(KE_total) #sorted array of ds
arr1inds = KE_total.isel(time=test_ind).values.argsort() # pick out indexes of sorted ds
plt.plot(np.arange(0,499,1),KE_total.isel(time=test_ind).values[arr1inds[::-1]],color='k')
plt.scatter(np.arange(0,499,1),KE_pred_total[arr1inds[::-1]],marker='.')
#plt.scatter(np.arange(0,499,1),KE_pred_total,marker='.')

