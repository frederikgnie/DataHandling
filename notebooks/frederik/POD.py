#%%
import xarray as xr
import numpy as np
#%%
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/data.zarr")
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
print('Creating numpy array of ds')
ds_np = np.stack((ds['u_vel'].values,ds['v_vel'].values,ds['w_vel']),axis=-1)
#%%
array = ds_np
n_snapshots = 4999
#Subtract mean in time
mean_snapshot = array.mean(axis=0)
for j in range(0,n_snapshots):
    array[j,:,:,:]=array[j,:,:,:] - mean_snapshot
array = np.reshape(array, (4999,-1)) #shape (4999,98304)
array = array.T
#array = array[:,:1000]
#array = array[:2,:]
#%% Economy SVD while n>m, and s have max m values.
u, s, vh = np.linalg.svd(array,full_matrices=False) #5,5min
np.save("/home/au569913/DataHandling/u",u)
np.save("/home/au569913/DataHandling/s",s)
np.save("/home/au569913/DataHandling/vh",vh)
np.save("/home/au569913/DataHandling/array",array)
np.save("/home/au569913/DataHandling/mean_snapshot",mean_snapshot)

#%%
import numpy as np
mean_snapshot = np.load("/home/au569913/DataHandling/mean_snapshot.npy")
array = np.load("/home/au569913/DataHandling/array.npy")
u = np.load("/home/au569913/DataHandling/u.npy")
s = np.load("/home/au569913/DataHandling/s.npy")
vh = np.load("/home/au569913/DataHandling/vh.npy")
#%%
firstmode = u[:,0].reshape(32,32,32,3)
secondmode =u[:,1].reshape(32,32,32,3)
u_vel=secondmode[:,:,:,0]
#%%
import matplotlib.pyplot as plt
from DataHandling import plots

uout = ds.copy() #copy test xr dataset
uout = uout.u_vel.isel(time=0)
uout.load()
#uout[:,:,:] = u_vel
uout[:,:,:] = c[:,:,:,0]
#uout[:,:,:] = mean_snapshot[:,:,:,0]

uout = uout.isel(z=16) #pick one time and z
fig = plt.figure()
ax = fig.add_subplot(111)
#air2d.T.plot(cmap='jet',vmin=0)
uout.T.plot.contourf(ax=ax,levels=200,cmap='jet')
ax.set_aspect('equal')

#data = uout.values #pick vel field
data = u_vel
plots.isocon(data,ds,'Target')
#%%
plt.scatter(range(0,len(s[0:200])),s[0:200],marker='.')
#%%
r = 200
#Recreate image from first r eigenvector
#test = np.matmul(np.matmul(u[:,0:r],u[:,0:r].T),array[:,0])

a = u[:,0:r].T @ array[:,0] # calculate r weigths
b = u[:,0:r] @ a # recontruct img with weights to eigenvectors
c = b.reshape(32,32,32,3) + mean_snapshot

x_test = array[:,0].reshape(32,32,32,3) + mean_snapshot
import importlib
importlib.reload(plots)
plots.uslice(c[:,16,:,0].T,x_test[:,16,:,0].T,'d',ds,'POD')

#%%
#if shapshots is rows first should be transposed. Since my snapshots are coluns we do reverse.
C = np.dot(array[:,0:n_snapshots], array[:,0:n_snapshots].T) / n_snapshots