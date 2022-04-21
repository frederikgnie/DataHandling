#%%
from DataHandling.features import preprocess
import xarray as xr
import numpy as np
domain = 'blonigan'
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura

#%% Execute SVD %%%
import numpy as np
from DataHandling import POD
#POD.POD(ds,domain)

#%% Load SVD  %%%%
import numpy as np
from DataHandling import POD
mean_snapshot = np.load("/home/au569913/DataHandling/models/POD/{}/mean_snapshot.npy".format(domain))
test_snap = np.load("/home/au569913/DataHandling/models/POD/{}/test_snap.npy".format(domain))
u = np.load("/home/au569913/DataHandling/models/POD/{}/u.npy".format(domain))
s = np.load("/home/au569913/DataHandling/models/POD/{}/s.npy".format(domain))
vh = np.load("/home/au569913/DataHandling/models/POD/{}/vh.npy".format(domain))

#%% Project ###
import importlib
importlib.reload(POD)
modes = [1536, 192, 24]
if domain == '1pi':
    modes = [3072,384,48]

c1,d1 = POD.projectPOD(modes[0],domain)

c2,d2 = POD.projectPOD(modes[1],domain)

c3,d3 = POD.projectPOD(modes[2],domain)
#%% Plot rms
from DataHandling import plots
importlib.reload(plots)

plots.rmsplot('POD',test_snap,c1,c2,c3,ds,domain)
#%% Plot energy of modes
import importlib
importlib.reload(POD)
POD.modeenergyplot(domain)

#%% Explore modes mean etc.
# Pick out some modes
firstmode = u[:,0].reshape(mean_snapshot.shape)
secondmode =u[:,1].reshape(mean_snapshot.shape)
seventhmode = u[:,7].reshape(mean_snapshot.shape)
forteenmode = u[:,14].reshape(mean_snapshot.shape)
u_vel1=firstmode[:,:,:,0]
u_vel2=secondmode[:,:,:,0]
# plot the picked out mode
import matplotlib.pyplot as plt
from DataHandling import plots

data = u_vel1
plots.isocon(data,ds,'PODmode1',domain,'uvel')
data = u_vel2
plots.isocon(data,ds,'PODmode2',domain,'uvel')
data = seventhmode[:,:,:,0]
plots.isocon(data,ds,'PODmode7',domain,'uvel')
data = forteenmode[:,:,:,0]
plots.isocon(data,ds,'PODmode14',domain,'uvel')

#%%
uout = ds.copy() #copy test xr dataset
uout = uout.u_vel.isel(time=0) #pick one time
uout.load()
uout[:,:,:] = u_vel1 
#uout[:,:,:] = c1[0,:,:,:,0] #pick out u_vle
#uout[:,:,:] = mean_snapshot[:,:,:,0]

uout = uout.isel(z=16) #pick one z
fig = plt.figure()
ax = fig.add_subplot(111)
#air2d.T.plot(cmap='jet',vmin=0)
uout.T.plot.contourf(ax=ax,levels=200,cmap='jet')
ax.set_aspect('equal')


#%% Uslice
#x_test is original data
x_test = test_snap + mean_snapshot
import importlib
from DataHandling import plots
importlib.reload(plots)
u_tau = 0.05
plots.uslice(c3/u_tau,test_snap/u_tau,'POD',domain,'z')
plots.uslice(c3/u_tau,test_snap/u_tau,'POD',domain,'y')

#Velocity fields (nofluc and nondim)
#plots.uslice(d3,x_test,'POD',domain,'z',save=False)
#plots.uslice(d3,x_test,'POD',domain,'y',save=False)

#%% Isocon For target/pred of test_ind
from DataHandling import postprocess
importlib.reload(plots)
u_tau=0.05
test_ind_toplot = 1 #30   #test_ind[208]=ind number 305 in time = time 3918 

data = postprocess.Qcrit(c3,ds,test_ind_toplot) # Calc q-criterion
plots.isocon(data,ds,'POD prediction',domain,'Qcrit')
# %% Error
from DataHandling import postprocess
import importlib
importlib.reload(postprocess)
#AE_error = postprocess.errornorm(predctions[0],target_list[0])
POD_error200 = postprocess.errornorm(c3,test_snap)

