#%%
from DataHandling.features import preprocess
import xarray as xr
import numpy as np
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/data.zarr")
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
#%% Execute SVD
import numpy as np
from DataHandling import POD
domain = 'nakamura'
#POD.POD(ds,domain)
#%% Load SVD
import numpy as np
from DataHandling import POD
domain = 'nakamura'
mean_snapshot = np.load("/home/au569913/DataHandling/models/POD/{}/mean_snapshot.npy".format(domain))
test_snap = np.load("/home/au569913/DataHandling/models/POD/{}/test_snap.npy".format(domain))
u = np.load("/home/au569913/DataHandling/models/POD/{}/u.npy".format(domain))
s = np.load("/home/au569913/DataHandling/models/POD/{}/s.npy".format(domain))
vh = np.load("/home/au569913/DataHandling/models/POD/{}/vh.npy".format(domain))

#%% Project   ###
import importlib
importlib.reload(POD)
modes = 1536
c1,d1 = POD.projectPOD(modes,domain)
modes = 192
c2,d2 = POD.projectPOD(modes,domain)
modes = 24
c3,d3 = POD.projectPOD(modes,domain)
#%% Plot rms
from DataHandling import plots
importlib.reload(plots)

plots.rmsplot('POD',test_snap,c1,c2,c3,ds,domain)

#%% Explore modes mean etc.
# Pick out some modes
firstmode = u[:,0].reshape(32,32,32,3)
secondmode =u[:,1].reshape(32,32,32,3)
u_vel1=firstmode[:,:,:,0]
u_vel2=secondmode[:,:,:,0]
# plot the picked out mode
import matplotlib.pyplot as plt
from DataHandling import plots
data = u_vel1
plots.isocon(data,ds,'PODmode1',domain,'uvel')
data = u_vel2
plots.isocon(data,ds,'PODmode2',domain,'uvel')

#
uout = ds.copy() #copy test xr dataset
uout = uout.u_vel.isel(time=0) #pick one time
uout.load()
#uout[:,:,:] = u_vel 
uout[:,:,:] = c1[0,:,:,:,0] #pick out u_vle
#uout[:,:,:] = mean_snapshot[:,:,:,0]

uout = uout.isel(z=16) #pick one z
fig = plt.figure()
ax = fig.add_subplot(111)
#air2d.T.plot(cmap='jet',vmin=0)
uout.T.plot.contourf(ax=ax,levels=200,cmap='jet')
ax.set_aspect('equal')

#data = uout.values #pick vel field
#plots.isocon(data,ds,'PODmode1',domain,'uvel')



#%% Plot energy of modes
import importlib
importlib.reload(POD)
POD.modeenergyplot(domain)

#%% Uslice
#x_test is original data
x_test = test_snap[10,:,:,:,:] + mean_snapshot
import importlib
importlib.reload(plots)
plots.uslice(d1[10,:,16,:,0].T,x_test[:,16,:,0].T,'d',ds,'POD')


# %% Kinetic energy
KE_total=postprocess.KE_ds(ds)
data = d1
KE_pred_total=postprocess.KE_np(data,ds)
#%%
KE_total[0:1000].plot(color='k',lw=0.5)
plt.scatter(np.arange(3000,6000,3),KE_pred_total/(0.05**2),marker='.')
# %% Error
from DataHandling import postprocess
import importlib
importlib.reload(postprocess)
#AE_error = postprocess.errornorm(predctions[0],target_list[0])
POD_error200 = postprocess.errornorm(c,test_snap)

# %% 

#%% kinetic energy of modes
from DataHandling import postprocess

modes = u.T.reshape(len(u[1]),32,32,32,3)
KE_modes = postprocess.KE_np(modes,ds)
#%%  plot kinetic energy of modes
plt.plot(np.arange(0,len(KE_modes[0:200])),KE_modes[0:200],marker='.')
#%%
#S = train_snap.T@ train_snap
#sigma, V = np.linalg.eig(S) 
U = train_snap @ V @ np.linalg.inv(sigma)