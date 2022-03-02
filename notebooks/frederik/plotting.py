#%%
import os
from DataHandling import utility
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
import importlib
importlib.reload(DataHandling.plots)
plots.uslice(predctions,target_list,'hej',ds,'z')
# %%

