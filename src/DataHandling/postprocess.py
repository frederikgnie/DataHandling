def Qcrit(data,ds,snapshot=0): 
    """"Ouputs Qcrit array

    Args:
        data (nparray): np array of  field
        ds (xarray): xarray dataset for determining coordinates
    """
    import numpy as np
    import xarray as xr    
    coords = ['x','y','z']
    fields = ['u_vel','v_vel','w_vel']
    #only one timestep
    #data=ds.isel(time=snapshot) #used if working
    data=data[snapshot,:,:,:,:]
    x = ds.coords['x'].values
    y = ds.coords['y'].values
    z = ds.coords['z'].values
    codict={'x':x,'y':y,'z':z}

    strtens =np.zeros(shape=(3,3,32,32,32))
    for i,fi in enumerate(fields):
        for j,co in enumerate(coords):
            #strtens[i,j,:,:,:] = data[fi].differentiate(co).values
            strtens[i,j,:,:,:] = np.gradient(data[:,:,:,i],codict[co],axis=j)
            
    S = 0.5*(strtens+strtens.transpose(1,0,2,3,4)) #Transpose strain tensor for all points
    A = 0.5*(strtens-strtens.transpose(1,0,2,3,4))
    Q = 0.5*(np.linalg.norm(A,axis=(0,1),ord=2)-np.linalg.norm(S,axis=(0,1),ord=2))
    return Q
# %%
def errornorm(pred,targ):
    import numpy as np
    error = np.linalg.norm(targ-pred)/np.linalg.norm(targ)
    return error

def KE_ds(ds):
    """Calculate kinetic energy for ds

    Args: 
        
    Returns: 
        KE_total(ds): shape (4999)
        
    """
    import xarray as xr
    import numpy as np
    # pick out coords array
    x = ds.coords['x'].values
    y = ds.coords['y'].values
    z = ds.coords['z'].values
    #ds = ds.reindex(y=ds.y[::-1])
    ds = ds.reindex(y=list(reversed(ds.y))) #to not get negative integration values for y
    # Map all inner product 
    ds = ds.assign(KE=lambda ds: 0.5*(ds['u_vel']*ds['u_vel']+ds['v_vel']*ds['v_vel']+ds['w_vel']*ds['w_vel']))
    ds = ds['KE'] #pick out only KE data array
    ds =ds.chunk('auto')
    ds = ds.integrate('y')
    ds = ds.integrate('x')
    ds = ds.integrate('z')
    print('Loading')
    ds = ds.load() #convert to np array
    KE_total = ds
    return KE_total

def KE_np(data,ds):
    """Calculate kinetic energy for numpy array

    Args: 
        data(np)
        ds(xrray): for coords
        
    Returns: 
        KE_pred_total(np): shape = data
        
    """
    import xarray as xr
    import numpy as np

    # pick out coords array
    x = ds.coords['x'].values
    y = ds.coords['y'].values
    z = ds.coords['z'].values

    KE_pred = np.zeros(shape=(np.shape(data[:,0,0,0,0])[0],32,32,32)) #initialise

    for l in range(np.shape(KE_pred[:,0,0,0])[0]):
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    KE_pred[l,i,j,k] = 0.5*(data[l,i,j,k,0]*data[l,i,j,k,0]+data[l,i,j,k,1]*data[l,i,j,k,1]+data[l,i,j,k,2]*data[l,i,j,k,2])
    #%%
    #integrate
    KE_pred_total = np.trapz(KE_pred,x,axis=1)
    KE_pred_total = np.trapz(KE_pred_total,z,axis=2)
    KE_pred_total = -np.trapz(KE_pred_total,y,axis=1)
    return KE_pred_total