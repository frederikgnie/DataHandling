def POD(ds):
    """Saves POD by SVD

    Args:
        
    Returns:
        
    """
    import numpy as np
    import xarray as xr
    print('Creating numpy array of ds')
    #ds_np = np.stack((ds['u_vel'].values,ds['v_vel'].values,ds['w_vel']),axis=-1)
    ds_np =np.load("/home/au569913/DataHandling/ds_np.npy")

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
    train_ind =np.load("/home/au569913/DataHandling/data/interim/train_ind.npy")
    test_snap=ds_np[test_ind]
    train_snap = ds_np[train_ind]

    #Subtract mean in time of training snapshots
    mean_snapshot = train_snap.mean(axis=0) 
    for j in range(0, len(train_ind)):
        train_snap[j,:,:,:]=train_snap[j,:,:,:] - mean_snapshot
    for j in range(0, len(test_ind)):
        test_snap[j,:,:,:]=test_snap[j,:,:,:] - mean_snapshot
    
    train_snap = np.reshape(train_snap, (len(train_ind),-1)) #shape (4999,98304)
    train_snap = train_snap.T #transpose for SVD #shape (98304,4999)


    #%% Economy SVD while n>m, and s have max m values.
    print('Performing SVD')
    u, s, vh = np.linalg.svd(train_snap,full_matrices=False) #5,5min
    print('Save results')
    
    np.save("/home/au569913/DataHandling/models/POD/u",u)
    np.save("/home/au569913/DataHandling/models/POD/s",s)
    np.save("/home/au569913/DataHandling/models/POD/vh",vh)
    np.save("/home/au569913/DataHandling/models/POD/test_snap",test_snap)
    np.save("/home/au569913/DataHandling/models/POD/mean_snapshot",mean_snapshot)

def projectPOD(modes):
    """Project POD modes 

    Args:
        
    Returns:
    c (nparray): Snapshots without mean

    """
    import numpy as np
    u = np.load("/home/au569913/DataHandling/u.npy")
    mean_snapshot = np.load("/home/au569913/DataHandling/mean_snapshot.npy")
    
    test_snap = np.load("/home/au569913/DataHandling/models/POD/test_snap.npy")
    ss = len(test_snap) # number of snapshots recreate from projection

    #Reshape to be compatible with matrix multi
    test_snap = np.reshape(test_snap, (len(test_snap),-1)) #shape (4999,98304)
    test_snap = test_snap.T #transpose for SVD #shape (98304,4999)

    r = modes #number of mode
    
    #test = np.matmul(np.matmul(u[:,0:r],u[:,0:r].T),array[:,0]) #alternative thats not working
    # calculate first ss snapshots
    a = u[:,0:r].T @ test_snap[:,0:ss] #calculate r weigths
    b = u[:,0:r] @ a # recontruct img with weights to eigenvectors
    c = b.T.reshape(ss,32,32,32,3) #fluctuations since no mean is added
    d = c + mean_snapshot #(4999,32,32,32,3) + (32,32,32,3)broadcasting
    return c,d

#Potential new based on correlation matrix
#if shapshots is rows first should be transposed. Since my snapshots are coluns we do reverse.
#C = np.dot(array[:,0:n_snapshots], array[:,0:n_snapshots].T) / n_snapshots