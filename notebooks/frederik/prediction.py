#%%
#25-02-2022
import os
import matplotlib
import importlib
from tensorflow import keras

import numpy as np
from glob import glob
from DataHandling import utility
from DataHandling import plots
from zipfile import BadZipfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from DataHandling.models import predict
#%%
tf_records = False

name="good-resonance-27"
overwrite=False
vars=['u_vel','v_vel','w_vel']
target = ['u_tar','v_tar','w_tar']
normalize=False
model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(name))
y_plus=15



#%% Thor style
if tf_records == True:
    

    pred = predict.predict(name,overwrite,model,y_plus,vars,target,normalize)


# %% fgn style
if tf_records == False:
    #Load data from xarray
    from DataHandling.features import preprocess 
    import xarray as xr
    from DataHandling import utility
    ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/data.zarr")
    ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
    #train_ind, validation_ind, test_ind = preprocess.split_test_train_val(ds) #find indexes
    train_ind=np.load("/home/au569913/DataHandling/data/interim/train_ind.npy")
    validation_ind =np.load("/home/au569913/DataHandling/data/interim/valid_ind.npy")
    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
    train = ds.isel(time = train_ind)
    valid = ds.isel(time = validation_ind)
    test = ds.isel(time = test_ind)

    #Convert to np array
    train_np=np.stack((train['u_vel'].values,train['v_vel'].values,train['w_vel']),axis=-1) #use values to pick data as np array and stack that shit
    valid_np=np.stack((valid['u_vel'].values,valid['v_vel'].values,valid['w_vel']),axis=-1)
    test_np=np.stack((test['u_vel'].values,test['v_vel'].values,test['w_vel']),axis=-1)
    
    #%% Predict
    predctions=[]
    print('<---Predicting now--->')
    predctions.append(model.predict(train_np,verbose=1))
    predctions.append(model.predict(valid_np,verbose=1))
    predctions.append(model.predict(test_np,verbose=1))

    #Using same targets as features
    target_list = [train_np,valid_np,test_np]

    print('Saving compressed arrays')
    _,output_path=utility.model_output_paths(name,y_plus,vars,target,normalize)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.savez_compressed(os.path.join(output_path,"predictions"),train=predctions[0],val=predctions[1],test=predctions[2])
    np.savez_compressed(os.path.join(output_path,"targets"),train=target_list[0],val=target_list[1],test=target_list[2])
    print("Saved data",flush=True)
    

# # %% plot figure
# uin = test.u_vel.isel(time=200,z=16)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# #air2d.T.plot(cmap='jet',vmin=0)
# uin.T.plot.contourf(ax=ax,levels=200,cmap='jet',vmin=0)
# ax.set_aspect('equal')

# #%%
# uout = test.copy() #copy test xr dataset
# uout['u_vel'].values = pred[:,:,:,:,0] #replace values with predictions
# uout = uout.u_vel.isel(time=200,z=16) #pick one time and z
# fig = plt.figure()
# ax = fig.add_subplot(111)
# #air2d.T.plot(cmap='jet',vmin=0)
# uout.T.plot.contourf(ax=ax,levels=200,cmap='jet',vmin=0)
# ax.set_aspect('equal')


# %%
