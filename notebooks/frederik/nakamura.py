#23-02-2022
#Based on no-sep-heat notebook this nb will try to incooperate nakamura network based on both tfrecord data base
#as well as a np array approach.
#%%
import os
import numpy as np
from pyexpat import model
from tensorflow import keras
from keras import layers
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from DataHandling.features import preprocess, slices
from DataHandling import utility
from DataHandling.models import models

os.environ['WANDB_DISABLE_CODE']='True'

wandbnotes = "Nakamura2, relu, plus_fluctuations"
tf_records = False

y_plus=15
repeat=3
shuffle=100
#batch_size=10
batch_size=100
activation='relu'
optimizer="adam"
loss='mean_squared_error'
patience=50
var=['u_vel','v_vel','w_vel']
target=['u_tar','v_tar','w_tar']
loss_dict={target[0]:'mean_squared_error',target[1]:'mean_squared_error',target[2]:'mean_squared_error'}
normalized=False
dropout=False
skip=4
model_type="baseline"

#%%Load data from tf scracth approach:
if tf_records == True:
    data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)
    train=data[0]
    validation=data[1]

#%% Load data from xarray approach:
if tf_records == False:
    import xarray as xr
    ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/data.zarr")
    ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
    u_tau = 0.05
    ds = preprocess.flucds(ds)/u_tau
    #train_ind, validation_ind, test_ind = preprocess.split_test_train_val(ds) #find indexes
    train_ind=np.load("/home/au569913/DataHandling/data/interim/train_ind.npy")
    validation_ind =np.load("/home/au569913/DataHandling/data/interim/valid_ind.npy")
    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

    train = ds.isel(time = train_ind)
    validation = ds.isel(time = validation_ind)
    test = ds.isel(time = train_ind)

    #Convert to np array
    train=np.stack((train['u_vel'].values,train['v_vel'].values,train['w_vel']),axis=-1) #use values to pick data as np array and stack that shit
    validation = np.stack((validation['u_vel'].values,validation['v_vel'].values,validation['w_vel']),axis=-1)

#%% Model
model=models.nakamura2(var,target,tf_records,activation)


#%% Initialise WandB & run
wandb.init(project="Thesis",notes=wandbnotes)

config=wandb.config
config.y_plus=y_plus
config.repeat=repeat
config.shuffle=shuffle
config.batch_size=batch_size
config.activation=activation
config.optimizer=optimizer
config.loss=loss
config.patience=patience
config.variables=var
config.target=target
config.dropout=dropout
config.normalized=normalized
config.skip=skip
config.model=model_type

model.compile(loss=loss, optimizer=optimizer)

logdir, backupdir= utility.get_run_dir(wandb.run.name)

#Callbacks
backup_cb=tf.keras.callbacks.ModelCheckpoint(os.path.join(backupdir,'weights.{epoch:02d}'),save_best_only=False)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,restore_best_weights=True)

#Model fit
if tf_records == True:  
#original , epochs=10000
    model.fit(x=train,epochs=10,validation_data=validation,callbacks=[WandbCallback(),early_stopping_cb,backup_cb])

#fgn version which utlisized format of xarray to np array
if tf_records == False:
    model.fit(x=train,y=train,epochs=300,validation_data=[validation, validation],callbacks=[WandbCallback(),early_stopping_cb,backup_cb])

#Model save
model.save(os.path.join("/home/au569913/DataHandling/models/trained",wandb.run.name))
print('Finished nakamura.py')