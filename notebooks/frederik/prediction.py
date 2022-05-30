#%%
#25-02-2022
# Following notebook will save predictions (and targets as well) of the corrosponding models.
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
#%% Inputs ###
tf_records = False
name = 'swift-sky-34'
model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(name))
domain = 'blonigan'
model_type = 'CNNAE' 

overwrite = False
vars=['u_vel','v_vel','w_vel']
target = ['u_tar','v_tar','w_tar']
normalize=False
y_plus=15

print('Running prediction.py with model: '+ name)
#%% Thor style
if tf_records == True:

    pred = predict.predict(name,overwrite,model,y_plus,vars,target,normalize)

# %% fgn style
if tf_records == False:

    pred = predict.predictxr(name, model, domain, model_type) #needs to have correct zarr/index save
    
    
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
