#%% Checking for weight behavior
model_type = 'SCAE'
domain = 'nakamura'
from tensorflow import keras
import numpy as np
#name = "warm-plasma-41" #l1=1e-6
#name = "mild-pine-42" #l1=1e-5
#name = 'fearless-shadow-54' #l1=1e-3
#name = 'curious-violet-69' #l1=e-2
#name = 'dulcet-armadillo-70' #l1=1e-1
#name = 'fragrant-flower-71' #l1=1 
name = 'dandy-bush-79' #l1=1e1
#name = 'northern-capybara-82' #l1=1e2
#name = 'celestial-dream-106'
#name = 'misty-night-98'
l1 = 1e1
model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(name))

# %% 
targ = np.load('/home/au569913/DataHandling/models/output/{}/targets.npz'.format(name))   
target=targ["test"] 
pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(name))
pred=pred["test"]
comp = np.load('/home/au569913/DataHandling/models/output/{}/comp.npz'.format(name))
comp = comp['test']
modecontent = comp.sum(axis=0).sum(axis=0).sum(axis=0).sum(axis=0) #shape 12
#comp = comp['comp']
#%%
#np.nonzero(comp[1,:,:,:,:])
import matplotlib.pyplot as plt
lan_var = []
for i in range(len(comp)): #499
    lan_var.append(np.count_nonzero(comp[i,:,:,:,:])) #last 0/1/2 seems to be the same 18851ish
plt.plot(np.arange(0,499,1),lan_var,color='k') # plot lantent variables
from statistics import mean, median 
mean_var = mean(lan_var)
median_var = median(lan_var)
#%%
from DataHandling import plots
import importlib
importlib.reload(plots)
meadian_var = plots.SCAElatent(comp,domain,name,save=True)
#%%
import DataHandling
from DataHandling import plots
import importlib
importlib.reload(plots)
#plots.uslice(pred,target,model_type,domain,'y',save=False)

plots.scaemode(comp,l1,domain,'z',save=False)

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
importlib.reload(plots)
data = comp[0,:,:,:,0]
plots.isocon(data,ds,'SCAEmode1',domain,'Qcrit',save=False)
data = comp[0,:,:,:,1]
plots.isocon(data,ds,'SCAEmode2',domain,'uvel',save=False)
data = comp[0,:,:,:,2]
plots.isocon(data,ds,'SCAEmode3',domain,'uvel',save=False)


# %% For manual comp
from DataHandling.features import preprocess
import xarray as xr
import numpy as np
domain = '1pi'
ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
u_tau = 0.05
ds = preprocess.flucds(ds)/u_tau
#train_ind, validation_ind, test_ind = preprocess.split_test_train_val(ds) #find indexes

test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

test = ds.isel(time = test_ind)
#Convert to np array
print('Converting to numpy array')
test_np=np.stack((test['u_vel'].values,test['v_vel'].values,test['w_vel']),axis=-1)
#%%
#manual comp 
from tensorflow import keras
import os
name = "celestial-dream-106"
model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(name))
keras_function = keras.backend.function([model.input], [model.layers[4].output])
comp_train = np.zeros(5)
comp_valid = np.zeros(5)
        
comp_test = keras_function([test_np])[0] #shape (499,32,32,32,3)
output_path ="/home/au569913/DataHandling/models/output/{}".format(name)
np.savez_compressed(os.path.join(output_path,"comp"),train=comp_train,val=comp_valid,test=comp_test)
# %%
