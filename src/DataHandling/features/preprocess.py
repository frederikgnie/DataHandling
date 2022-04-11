def save_tf(y_plus,var,target,data,normalized=False):
    """Takes a xarray dataset extracts the variables in var and saves them as a tfrecord

    Args:
        y_plus (int): at which y_plus to take a slice
        var (list): list of inputs to save. NOT with target
        target (list): list of target. Only 1 target for now
        data (xarray): dataset of type xarray
        normalized(bool): if the data is normalized or not

    Returns:
        None:
    """

    import os
    import xarray as xr
    import numpy as np
    import dask
    import tensorflow as tf
    from DataHandling import utility
    import shutil
    import json

    def custom_optimize(dsk, keys):
        dsk = dask.optimization.inline(dsk, inline_constants=True)
        return dask.array.optimization.optimize(dsk, keys)



    def numpy_to_feature(numpy_array):
        """Takes an numpy array and returns a tf feature

        Args:
            numpy_array (ndarray): numpy array to convert to tf feature

        Returns:
            Feature: Feature object to use in an tf example
        """
        feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(numpy_array)).numpy()]))
        return feature



    def serialize(slice_array,var):
        """Constructs an serialzied tf.Example package

        Args:
            slice_array (xarray): A xaray
            var (list): a list of the variables that are to be serialized

        Returns:
            protostring: protostring of tf.train.Example
        """

        feature_dict={}
        for name in var:
            feature=slice_array[name].values
            if type(feature) is np.ndarray:
                feature_dict[name] = numpy_to_feature(feature)
            else:
                raise Exception("other inputs that xarray/ numpy has not yet been defined")
        
        proto=tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return proto.SerializeToString()


    def split_test_train_val(slice_array,test_split=0.1,validation_split=0.2):
        """Splits the data into train,test,val

        Args:
            slice_array (xarray): The sliced data to be split
            test_split (float, optional): the test split. Defaults to 0.1.
            validation_split (float, optional): the validation split. Defaults to 0.2.

        Returns:
            tuple: returns the selected indices for the train, validation,test split
        """
        num_snapshots=len(slice_array['time'])
        train=np.arange(0,num_snapshots)
        validation=np.random.choice(train,size=int(num_snapshots*validation_split),replace=False)
        train=np.setdiff1d(train,validation)
        test=np.random.choice(train,size=int(num_snapshots*test_split),replace=False)
        train=np.setdiff1d(train,test)
        np.random.shuffle(train)

        return train, validation, test



    def save_load_dict(var,save_loc):
        """Saves an json file with the file format. Makes it possible to read the data back again

        Args:
            var (list): list of variables to include
        """
        load_dict={}

        for name in var:
            load_dict[name] = "array_serial"

        with open(os.path.join(save_loc,'format.json'), 'w') as outfile:
            json.dump(load_dict,outfile)


    client, cluster =utility.slurm_q64(1,time='0-01:30:00',ram='50GB')

    #Define save location to save data
    save_loc=slice_loc(y_plus,var,target,normalized)
    
    #//fgn: trying without appending the "target" as doing autoencoder
    #append the target
    #var.append(target[0])
    
    for i,j in enumerate(target): 
        var.append(j)



    #select y_plus value and remove unessary components. Normalize if needed
    
    #Re = 10400 #Direct from simulation
    #nu = 1/Re #Kinematic viscosity
#
    #slice_array=data
#
    #if target[0]=='tau_wall':
    #    target_slice1=slice_array['u_vel'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
    #    target_slice1=nu*target_slice1
    #    
    #    #target_slice2=slice_array['u_vel'].differentiate('y').sel(y=slice_array['y'].max(),method="nearest")
    #    #target_slice2=nu*target_slice2
    #    
    #    if normalized==True:
    #        target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))
    #
    ##Checking if the target contains _flux
    #elif target[0][-5:] =='_flux':
    #    target_slice1=slice_array[target[0][:-5]].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
    #    pr_number=float(target[0][2:-5])
    #    target_slice1=nu/(pr_number)*target_slice1
    #    
    #    #target_slice2=slice_array[target[0][:-5]].differentiate('y').sel(y=slice_array['y'].max(),method="nearest")
    #    #target_slice2=nu/(pr_number)*target_slice2
    #    
    #    if normalized==True:
    #        target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))
    #else:
    #    target_slice1=slice_array[target[0]].sel(y=utility.y_plus_to_y(0),method="nearest")
#
    #    #target_slice2=slice_array[target[0]].sel(y=slice_array['y'].max(),method="nearest")
    #    if normalized==True:
    #        target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))
#
#
    #other_wall_y_plus=utility.y_to_y_plus(slice_array['y'].max())-y_plus
    #
    #if normalized==True:
    #    slice_array=(slice_array-slice_array.mean(dim=('time','x','z')))/(slice_array.std(dim=('time','x','z')))

    
    
    #Dont slice //fgn
    wall_1 = data
    wall_1[target[0]] = wall_1[var[0]]
    wall_1[target[1]] = wall_1[var[1]]
    wall_1[target[2]] = wall_1[var[2]]
    wall_1=wall_1[var] # Picks out only var + target lidt e.g. u_vel & tau_wall

    #Slice
    #wall_1=slice_array.sel(y=utility.y_plus_to_y(y_plus),method="nearest")
    #wall_1[target[0]]=target_slice1 
    #wall_1=wall_1[var] # Picks out only var + target lidt e.g. u_vel & tau_wall
    
    #wall_2=slice_array.sel(y=utility.y_plus_to_y(other_wall_y_plus),method="nearest")
    #wall_2[target[0]]=target_slice2
    #wall_2=wall_2[var]
    
 
    #wall_1,wall_2=dask.compute(*[wall_1,wall_2])

    #shuffle the data, split into 3 parts and save
    train_1, validation_1, test_1 = split_test_train_val(wall_1)

    wall_1=wall_1.compute()
    

    #train_2, validation_2, test_2 = split_test_train_val(wall_2)

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    else:
        print('deleting old version')
        shutil.rmtree(save_loc)           
        os.makedirs(save_loc)

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    
    with tf.io.TFRecordWriter(os.path.join(save_loc,'train'),options) as writer:
        print('train',flush=True)
        for i in train_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in train_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)
        writer.close()


    with tf.io.TFRecordWriter(os.path.join(save_loc,'test'),options) as writer:
        print('test',flush=True)
        for i in test_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in test_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)
        writer.close()

    with tf.io.TFRecordWriter(os.path.join(save_loc,'validation'),options) as writer:
        print('validation',flush=True)
        for i in validation_1:
            write_d=serialize(wall_1.isel(time=i),var)
            writer.write(write_d)
        # for i in validation_2:
        #         write_d=serialize(wall_2.isel(time=i),var)
        #         writer.write(write_d)    
        writer.close()


    save_load_dict(var,save_loc)
    client.close()
    del wall_1
    return None


def slice_loc(y_plus,var,target,normalized):
    """where to save the slices

    Args:
        y_plus (int): y_plus value of slice
        var (list): list of variables
        target (list): list of targets
        normalized (bool): if the data is normalized or not

    Returns:
        str: string of file save location
    """
    import os

    var_sort=sorted(var)
    var_string="_".join(var_sort)
    target_sort=sorted(target)
    target_string="_".join(target_sort)

    if normalized==True:
        slice_loc=os.path.join("/home/au569913/DataHandling/data/processed",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string+"-normalized")
    else:
        slice_loc=os.path.join("/home/au569913/DataHandling/data/processed",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string)

    return slice_loc

# Adding following fuction from indside other function to try to use
def split_test_train_val(slice_array,test_split=0.1,validation_split=0.2):
    import numpy as np
    import xarray as xr
    """Splits the data into train,test,val
    Args:
        slice_array (xarray): The sliced data to be split
        test_split (float, optional): the test split. Defaults to 0.1.
        validation_split (float, optional): the validation split. Defaults to 0.2.
    Returns:
        tuple: returns the selected indices for the train, validation,test split
    """
    
    num_snapshots=len(slice_array['time'])
    train=np.arange(0,num_snapshots)
    validation=np.random.choice(train,size=int(num_snapshots*validation_split),replace=False)
    train=np.setdiff1d(train,validation)
    test=np.random.choice(train,size=int(num_snapshots*test_split),replace=False)
    train=np.setdiff1d(train,test)
    np.random.shuffle(train)
    np.save("/home/au569913/DataHandling/data/interim/train_ind",train)
    np.save("/home/au569913/DataHandling/data/interim/valid_ind",validation)
    np.save("/home/au569913/DataHandling/data/interim/test_ind",test)
    return train, validation, test

def standardize(ds):
    """Takes a xarray dataset and standardize

    Args:
        ds (xarray): dataset of type xarray

    Returns:
        ds (xarray): dataset of type xarray
    """
    ds=(ds-ds.mean(dim=('time','x','y','z')))/(ds.std(dim=('time','x','y','z')))
    return ds

def normalize(ds):
    """Takes a xarray dataset and normalize

    Args:
        ds (xarray): dataset of type xarray

    Returns:
        ds (xarray): dataset of type xarray
    """
    ds=ds/ds.max()
    return ds

def flucds(ds):
    #ds=ds-ds.mean(dim=('x','y','z')) #frederik implementation
    ds=ds-ds.mean(dim=('time','x','z'))
    return ds

def flucnp(array):
    """Calculates fluctuation based on nparray.

    Args:
        array (nparray): 

    Returns:
        array (nparray): 
    """
    import numpy as np
    import xarray as xr
    #Nakamura approach - do whole 
    #u_mean = array[:,:,:,:,0].mean(axis=3).mean(axis=0).mean(axis=0)
    #v_mean = array[:,:,:,:,1].mean(axis=3).mean(axis=0).mean(axis=0)
    #w_mean = array[:,:,:,:,2].mean(axis=3).mean(axis=0).mean(axis=0)

    mean = array.mean(axis=3).mean(axis=0).mean(axis=0)

    fluc = np.zeros(array.shape)
    for i in range(32):
        fluc[:,:,i,:,:] = array[:,:,i,:,:]-mean[i,:]
    
    return fluc

def rms(fluc):
    """Calculates rms of 3 vel componets w.r.t y axis

    Args:
        array (nparray): shape(time,x,y,z,vel)

    Returns:
        array (nparray): shape(y,vel)
    """
    import numpy as np
    import xarray as xr
    rms=np.sqrt(np.mean(fluc**2,axis=3).mean(axis=0).mean(axis=0))
    return rms

def testforconv(ds_np,batch):
    """Test for convergence of the simulation

    Args:
        ds (xarray): dataset of type xarray
        batch (int): size of batch to test on

    Returns:
        ds (xarray): dataset of type xarray
    """
    import xarray as xr
    import numpy as np
    ds_np=flucnp(ds_np)
    first = ds_np[0:batch,:,:,:,:]
    last = ds_np[-batch:-1,:,:,:,:]
    first = np.sqrt(np.mean(first**2,axis=3).mean(axis=0).mean(axis=0).mean(axis=0))
    last = np.sqrt(np.mean(last**2,axis=3).mean(axis=0).mean(axis=0).mean(axis=0))

    #ds approach
    #ds = flucds(ds)
    #first = ds.isel(time=slice(0,batch)).std(dim=('time','x','y','z'))
    #last = ds.isel(time=slice(-batch,-1)).std(dim=('time','x','y','z'))
    #first = first.load()
    #last = last.load()
    #first = [first.u_vel.values, first.v_vel.values, first.w_vel.values]
    #last = [last.u_vel.values, last.v_vel.values, last.w_vel.values]
    return first, last

#def savedsmean(ds,domain):
#    import numpy as np
#    import xarray as xr
#    ds_mean=ds.mean(dim=('time','x','z'))
#    mean_np=np.stack((ds_mean['u_vel'].values,ds_mean['v_vel'].values,ds_mean['w_vel']),axis=-1)
#    np.save("/home/au569913/DataHandling/data/interim/train_ind",meanfield_)
