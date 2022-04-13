#%% Checking for weight behavior
model_type = 'SCAE'
domain = 'blonigan'
from tensorflow import keras
import numpy as np
#name = "warm-plasma-41" #l1=1e-6
#name = "mild-pine-42" #l1=1e-5
name = 'fearless-shadow-54' #l1=1e-3
model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(name))
a= model.get_weights()
for i in range(len(a)):
    print(np.median(a[i]))


# %% 
comp = np.load('/home/au569913/DataHandling/models/output/{}/comp.npz'.format(name))
comp = comp['test']
#comp = comp['comp']
#%%
#np.nonzero(comp[1,:,:,:,:])
import matplotlib.pyplot as plt
lan_var = []
for i in range(len(comp)):
    lan_var.append(len(np.nonzero(comp[i,:,:,:,0])[0])) #last 0/1/2 seems to be the same 18851ish
plt.plot(np.arange(0,499,1),lan_var,color='k') # plot lantent variables
#%%
import DataHandling
from DataHandling import plots
import importlib
importlib.reload(plots)
#plots.uslice(comp,comp,model_type,domain,'y',save=False)
plots.scaemode(comp,model_type,domain,'y',save=False)
#%%
import xarray as xr
import numpy as np
domain = 'blonigan'
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
#%%
# plot the picked out mode
import matplotlib.pyplot as plt
from DataHandling import plots

data = comp[0,:,:,:,0]
plots.isocon(data,ds,'SCAEmode1',domain,'Qcrit')
data = comp[0,:,:,:,1]
plots.isocon(data,ds,'SCAEmode2',domain,'uvel')
data = comp[0,:,:,:,2]
plots.isocon(data,ds,'SCAEmode3',domain,'uvel')

