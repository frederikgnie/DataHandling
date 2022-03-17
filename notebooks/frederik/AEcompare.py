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
plots.rmsplot('AE',target,predlist[0],predlist[1],predlist[2],ds)

# %%
