def POD(ds,domain):
    """Saves POD by SVD

    Args:
    ds (xrray)
    domain (str)
        
    Returns:
        
    """
    import numpy as np
    import xarray as xr
    print('Creating numpy array of ds')
    ds_np = np.stack((ds['u_vel'].values,ds['v_vel'].values,ds['w_vel']),axis=-1)
    #ds_np =np.load("/home/au569913/DataHandling/ds_np.npy") #load if already created

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
    train_ind =np.load("/home/au569913/DataHandling/data/interim/train_ind.npy")
    test_snap=ds_np[test_ind]
    train_snap = ds_np[train_ind]

    #Subtract mean in time of training snapshots
    print('Subtracting mean')
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
    
    np.save("/home/au569913/DataHandling/models/POD/{}/u".format(domain),u)
    np.save("/home/au569913/DataHandling/models/POD/{}/s".format(domain),s)
    np.save("/home/au569913/DataHandling/models/POD/{}/vh".format(domain),vh)
    np.save("/home/au569913/DataHandling/models/POD/{}/test_snap".format(domain),test_snap)
    np.save("/home/au569913/DataHandling/models/POD/{}/mean_snapshot".format(domain),mean_snapshot)

def projectPOD(modes,domain):
    """Project POD modes 

    Args:
        
    Returns:
    c (nparray): Snapshots without mean

    """
    import numpy as np
    u = np.load("/home/au569913/DataHandling/models/POD/{}/u.npy".format(domain))
    mean_snapshot = np.load("/home/au569913/DataHandling/models/POD/{}/mean_snapshot.npy".format(domain))
    
    test_snap = np.load("/home/au569913/DataHandling/models/POD/{}/test_snap.npy".format(domain))
    ss = len(test_snap) # number of snapshots recreate from projection

    #Reshape to be compatible with matrix multi
    test_snap = np.reshape(test_snap, (len(test_snap),-1)) #shape (4999,98304)
    test_snap = test_snap.T #transpose for SVD #shape (98304,4999)

    r = modes #number of mode
    
    #test = np.matmul(np.matmul(u[:,0:r],u[:,0:r].T),array[:,0]) #alternative thats not working
    # calculate first ss snapshots
    a = u[:,0:r].T @ test_snap[:,0:ss] #calculate r weigths
    b = u[:,0:r] @ a # recontruct img with weights to eigenvectors
    c = b.T.reshape((ss,) + mean_snapshot.shape) #ss,32,32,32,3 fluctuations since no mean is added
    d = c + mean_snapshot #(4999,32,32,32,3) + (32,32,32,3)broadcasting
    return c,d

#Potential new based on correlation matrix
#if shapshots is rows first should be transposed. Since my snapshots are coluns we do reverse.
#C = np.dot(array[:,0:n_snapshots], array[:,0:n_snapshots].T) / n_snapshots
#%%
def modeenergyplot(domain):
    """Plot energy and cumulative energy of POD modes (s)
    Args:
    domain (str): The domain on which the POD has been carried out.
    Returns:

    """
    import numpy as np
    import matplotlib.pyplot as plt
    s = np.load("/home/au569913/DataHandling/models/POD/{}/s.npy".format(domain))
    s=s**2 # energy is singular values 
    cum_s = (s.cumsum()/s.sum())
    
    name = ''
    modes = 400
    labels = ['DNS',r'$r=1536$',r'$r=192$',r'$r=24$']

    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,2,figsize=([15*cm,5*cm]),sharex=False,sharey=False,constrained_layout=True,dpi=1000)
    
    axs[0].plot(range(0,len(s[0:modes])),s[0:modes],marker='.',lw=0.7,color='k', ms=2.5)
    axs[1].plot(range(0,len(s[0:modes])),cum_s[0:modes],marker='.',lw=0.7,color='k', ms=2.5)
    
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_yscale('log', base=10)
    
    #axs[0].set_title(name.capitalize(),weight="bold")
    axs[0].set_ylabel(r"Energy")
    axs[0].set_xlabel(r'POD mode') 
    
    axs[1].set_ylabel(r"Cumulative Energy")
    axs[1].set_xlabel(r'POD mode')

    #Setting labels and stuff
    plt.savefig("/home/au569913/DataHandling/reports/{}/modeenergy.pdf".format(domain),bbox_inches='tight')
    #plt.show()
