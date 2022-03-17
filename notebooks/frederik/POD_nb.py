#%%
from DataHandling.features import preprocess
import xarray as xr
import numpy as np
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/data.zarr")
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
#%% Load SVD
import numpy as np
mean_snapshot = np.load("/home/au569913/DataHandling/models/POD/mean_snapshot.npy")
test_snap = np.load("/home/au569913/DataHandling/models/POD/test_snap.npy")
u = np.load("/home/au569913/DataHandling/models/POD/u.npy")
s = np.load("/home/au569913/DataHandling/models/POD/s.npy")
vh = np.load("/home/au569913/DataHandling/models/POD/vh.npy")
#%% Project
from DataHandling import POD
import importlib
importlib.reload(POD)
modes = 1536
c1,d1 = POD.projectPOD(modes)
modes = 192
c2,d2 = POD.projectPOD(modes)
modes = 24
c3,d3 = POD.projectPOD(modes)
#%%
from DataHandling import plots
plots.rmsplot('POD',test_snap,c1,c2,c3,ds)

#%% Explore modes mean etc.
# Pick out some modes
firstmode = u[:,0].reshape(32,32,32,3)
secondmode =u[:,1].reshape(32,32,32,3)
u_vel=secondmode[:,:,:,0]
# plot the picked out mode
import matplotlib.pyplot as plt
from DataHandling import plots

uout = ds.copy() #copy test xr dataset
uout = uout.u_vel.isel(time=0) #pick one time
uout.load()
#uout[:,:,:] = u_vel 
uout[:,:,:] = c[0,:,:,:,0] #pick out u_vle
#uout[:,:,:] = mean_snapshot[:,:,:,0]

uout = uout.isel(z=16) #pick one z
fig = plt.figure()
ax = fig.add_subplot(111)
#air2d.T.plot(cmap='jet',vmin=0)
uout.T.plot.contourf(ax=ax,levels=200,cmap='jet')
ax.set_aspect('equal')

#data = uout.values #pick vel field
data = u_vel
plots.isocon(data,ds,'Target')
#%% Plot energy of modes
plt.scatter(range(0,len(s[0:200])),s[0:200],marker='.')

#%%
#x_test is from original data
#x_test = array[:,0].reshape(32,32,32,3) + mean_snapshot 
x_test = test_snap[10,:,:,:,:] + mean_snapshot
import importlib
importlib.reload(plots)
plots.uslice(d[10,:,16,:,0].T,x_test[:,16,:,0].T,'d',ds,'POD')

#%%
from DataHandling.features import preprocess
import matplotlib.pyplot as plt
rms_pred = preprocess.rms(c) 
rms_tar = preprocess.rms(test_snap)
u_tau = 0.05
y = ds.coords['y'].values
y = abs(y-2)
plt.plot(y,rms_tar[:,0]/u_tau)
plt.plot(y,rms_pred[:,0]/u_tau)
#plt.plot(y,rms_tar[:,1])
#plt.plot(y,rms_pred[:,1])
# %% Kinetic energy
KE_total=postprocess.KE_ds(ds)
data = d 
KE_pred_total=postprocess.KE_np(data,ds)
#%%
KE_total[0:1000].plot(color='k',lw=0.5)
plt.scatter(np.arange(3000,6000,3),KE_pred_total/(0.05**2),marker='.')
# %% Error
from DataHandling import postprocess
import importlib
importlib.reload(postprocess)
AE_error = postprocess.errornorm(predctions[0],target_list[0])
POD_error200 = postprocess.errornorm(c,test_snap)

# %%
