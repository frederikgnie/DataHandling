def predict(model_name,overwrite,model,y_plus,var,target,normalized):
    """Uses a trained model to predict with

    Args:
        model_name (str): the namen given to the model by Wandb
        overwrite (Bool): Overwrite existing data or not
        model (object): the loaded model
        y_plus (int): y_plus value
        var (list): the variabels used as input
        target (list): list of target
        normalized (Bool): If the model uses normalized data
    """
    import os
    from DataHandling import utility
    from DataHandling.features import slices
    import shutil
    import numpy as np
    
    _,output_path=utility.model_output_paths(model_name,y_plus,var,target,normalized)


    data_exist=False

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    elif os.path.exists(os.path.join(output_path,'targets.npz')) and overwrite==False:
        data_exist=True
        print("Data exists and overwrite is set to false. Exiting")

    elif os.path.exists(os.path.join(output_path,'targets.npz')) and overwrite==True:
        print("deleting folder",flush=True)
        shutil.rmtree(output_path)


    if data_exist==False:
        data=slices.load_validation(y_plus,var,target,normalized)
        feature_list=[]
        target_list=[]

        for data_type in data: #data_type: train, val, test 

            feature_list.append(data_type[0])

            target_list.append(data_type[1].numpy()) #turns tf.tensor to numpy array
            
        predctions=[]

        print('<---Predicting now--->')
        predctions.append(model.predict(feature_list[0]))
        predctions.append(model.predict(feature_list[1]))
        predctions.append(model.predict(feature_list[2]))

        #predctions=[np.squeeze(x,axis=3) for x in predctions]
        print('Saving compressed arrays')
        np.savez_compressed(os.path.join(output_path,"predictions"),train=predctions[0],val=predctions[1],test=predctions[2])
        np.savez_compressed(os.path.join(output_path,"targets"),train=target_list[0],val=target_list[1],test=target_list[2])


        print("Saved data",flush=True)

def predictxr(model_name, model, domain, network):
    """Uses a trained model to predict with. 
    NEED TO BE RUN WHILE zarr/index IS CURRENT model.
    This is because the target values are saved straight from xarray ds loading. 

    Args:
        model_name (str): the namen given to the model by Wandb
        overwrite (Bool): Overwrite existing data or not
        model (object): the loaded model
        normalized (Bool): If the model uses normalized data
    """
    #Load data from xarray
    from DataHandling.features import preprocess 
    import xarray as xr
    from DataHandling import utility
    import numpy as np
    import os
    from tensorflow import keras
    print('Loading ds and selecting train/val/test index')
    ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
    ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
    u_tau = 0.05
    ds = preprocess.flucds(ds)/u_tau
    #train_ind, validation_ind, test_ind = preprocess.split_test_train_val(ds) #find indexes
    train_ind=np.load("/home/au569913/DataHandling/data/interim/train_ind.npy")
    validation_ind =np.load("/home/au569913/DataHandling/data/interim/valid_ind.npy")
    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
    train = ds.isel(time = train_ind)
    valid = ds.isel(time = validation_ind)
    test = ds.isel(time = test_ind)

    #Convert to np array
    print('Converting to numpy array')
    train_np=np.stack((train['u_vel'].values,train['v_vel'].values,train['w_vel']),axis=-1) #use values to pick data as np array and stack that shit
    valid_np=np.stack((valid['u_vel'].values,valid['v_vel'].values,valid['w_vel']),axis=-1)
    test_np=np.stack((test['u_vel'].values,test['v_vel'].values,test['w_vel']),axis=-1)
    
    #%% Predict
    predctions=[]
    print('<---Predicting now--->')
    batch_size = 32
    predctions.append(model.predict(train_np,verbose=1,batch_size=batch_size))
    predctions.append(model.predict(valid_np,verbose=1,batch_size=batch_size))
    predctions.append(model.predict(test_np,verbose=1,batch_size=batch_size))

    #Using same targets as features
    target_list = [train_np,valid_np,test_np]

    print('Saving compressed arrays')
    output_path ="/home/au569913/DataHandling/models/output/{}".format(model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.savez_compressed(os.path.join(output_path,"predictions"),train=predctions[0],val=predctions[1],test=predctions[2])
    np.savez_compressed(os.path.join(output_path,"targets"),train=target_list[0],val=target_list[1],test=target_list[2])
      
    if network == 'SCAE':
        #keras function can return intermediate layer output
        keras_function = keras.backend.function([model.input], [model.layers[4].output])
        comp_train = keras_function([train_np])[0] #shape (499,32,32,32,12)
        comp_valid = keras_function([valid_np])[0] #shape (499,32,32,32,12)
        comp_test = keras_function([test_np])[0] #shape (499,32,32,32,12)
        np.savez_compressed(os.path.join(output_path,"comp"),train=comp_train,val=comp_valid,test=comp_test)
    if network == 'CNNAE' and model_name == 'cosmic-feather-29': #Nakamura8
        keras_function = keras.backend.function([model.input], [model.layers[24].output])
        #comp_train = keras_function([train_np])[0] #shape ()
        comp_train = np.zeros(5)
        comp_valid = np.zeros(5)
        #comp_valid = keras_function([valid_np])[0] #shape ()
        comp_test = keras_function([test_np])[0] #shape (499,32,32,32,3)

        np.savez_compressed(os.path.join(output_path,"comp"),train=comp_train,val=comp_valid,test=comp_test)
    print("Saved data",flush=True)
