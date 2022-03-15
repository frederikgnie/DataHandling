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

