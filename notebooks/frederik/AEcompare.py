#%%
import os
from DataHandling import plots, utility
from DataHandling.features import slices
import shutil
import numpy as np
from tensorflow import keras


# Load model prediction/target
name=["cosmic-feather-29","valiant-river-31","deep-leaf-32"]

predlist=[]
for i in name:
    model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(i))
    pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(i))
    predlist.append(pred["test"])
targ = np.load('/home/au569913/DataHandling/models/output/{}/targets.npz'.format(name[0]))   

target=targ["test"] 
# %% Plot rms
from DataHandling import plots
plots.rmsplot('AE',target,predlist[0],predlist[1],predlist[2],ds,'nakamura')

# %%
from DataHandling import postprocess
import importlib
importlib.reload(postprocess)
AE_error=[]
for i in range(0,3):
    AE_error.append(postprocess.errornorm(predlist[i],target))
POD_error = [postprocess.errornorm(c1,test_snap), postprocess.errornorm(c2,test_snap),postprocess.errornorm(c3,test_snap)]
