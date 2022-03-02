#%%

""" Bruges til at lave nye slices af dataen """

from DataHandling.features import slices, preprocess
import xarray as xr


df=xr.open_zarr("/home/au569913/DataHandling/data/interim/data.zarr")


### Original 
#var=['u_vel']
#target=['tau_wall']
#normalized=False
#y_plus=15
#slices.save_tf(y_plus,var,target,df,normalized=normalized)

### fgn tries this
df=df.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
var=['u_vel','v_vel','w_vel']
target=['u_tar','v_tar','w_tar']
normalized=False
y_plus=15
preprocess.save_tf(y_plus,var,target,df,normalized=normalized)

print('Saved tf_records')

# %%
