


def threeD_plot(error_val,output_path):
    """3d KDE of the errors

    Args:
        error_val (numpy array): the errors
        output_path (Path): where to save

    Returns:
        None: 
    """
    import numpy as np
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu


    train_numpy=error_val.to_numpy()
    num_snapshots=int(train_numpy.shape[0]/256/256)
    reshape_t=train_numpy.reshape((num_snapshots,256,256))
    avg=np.mean(reshape_t,0)

    # Create meshgrid
    xx, yy = np.mgrid[0:256:256j, 0:256:256j]


    x_range=12
    z_range=6

    gridpoints_x=int(255)+1
    gridponts_z=int(255)+1


    x_plus_max=x_range*u_tau/nu
    z_plus_max=z_range*u_tau/nu


    x_plus_max=np.round(x_plus_max).astype(int)
    z_plus_max=np.round(z_plus_max).astype(int)

    axis_range_x=np.array([0,950,1900,2850,3980,4740])
    axis_range_z=np.array([0,470,950,1420,1900,2370])


    placement_x=axis_range_x*nu/u_tau
    placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


    placement_z=axis_range_z*nu/u_tau
    placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)




    cm =1/2.54
    fig = plt.figure(figsize=(15*cm,10*cm),dpi=200)
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, np.transpose(avg), cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_zlabel(r'Error $\%$')
    ax.set_box_aspect((2,1,1))

    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x)
    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x)

    ax.set_yticks(placement_z)
    ax.set_yticklabels(axis_range_z)
    fig.colorbar(surf, shrink=0.3, aspect=5) # add color bar indicating the PDF
    ax.view_init(30, 140)
    
    fig.savefig(os.path.join(output_path,'validation_3D.pdf'),bbox_inches='tight')

    return None

def pdf_plots(error_fluc,names,output_path,target_type):
    """Makes both boxplot and pdf plot for the errors.

    Args:
        error_fluc (list): list of the train,validation,test errors in local form
        names (list): list of the names of the data. Normally train,validaiton,test
        output_path (Path): Path of where to save the figures
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import seaborn as sns
    import KDEpy
    


    sns.set_theme()
    sns.set_context("paper")
    sns.set_style("ticks")
    
    for i in range(3):
        cm =1/2.54
        fig, ax = plt.subplots(1, 1,figsize=(15*cm,10*cm),dpi=100)
        
        
        

        if target_type=="flux":
            sns.boxplot(data=error_fluc[i]['Root sq. error of local heat flux'],showfliers = False,orient='h',ax=ax)
        elif target_type=="stress":
            sns.boxplot(data=error_fluc[i]['Root sq. error of local shear stress'],showfliers = False,orient='h',ax=ax)
        
        sns.despine()
        fig.savefig(os.path.join(output_path,names[i]+'_boxplot.pdf'),bbox_inches='tight',format='pdf')
        plt.clf()
        
        fig, ax = plt.subplots(1, 1,figsize=(15*cm,10*cm),dpi=100)
        max_range_error=error_fluc[i].max().to_numpy().item()*1.1
        min_range_error=error_fluc[i].min().to_numpy().item()*0.99
        #Find coeffs to find the log equiv.

        x_grid = np.linspace(min_range_error, max_range_error, num=int(max_range_error)*2)
        y_fluct = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(error_fluc[i].to_numpy(), weights=None).evaluate(x_grid)
        

        sns.lineplot(x=x_grid, y=y_fluct,ax=ax)


        ax.set_xlabel(r'Error $\left[\% \right]$')
        ax.set_ylabel('Density')
        # if target_type=="flux":
        #     y_local = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(error_fluc[i]['Root sq. error of local heat flux'].to_numpy(), weights=None).evaluate(x_grid)
        #     sns.lineplot(x=x_grid, y=y_local, label='Root sq. error of local heat flux',ax=ax)
        # else:
        #     y_local = KDEpy.FFTKDE(bw='ISJ', kernel='gaussian').fit(error_fluc[i]['Root sq. error of local shear stress'].to_numpy(), weights=None).evaluate(x_grid)
        #     sns.lineplot(x=x_grid, y=y_local, label='Root sq. error of local shear stress',ax=ax)
        
        sns.despine()

        ax.fill_between(x_grid,y_fluct,alpha=0.8,color='grey')
        
        #ax.fill_between(x_grid,y_local,alpha=0.4,color='grey')
        #ax.set(xscale='log')
        ax.set_xlim(-1,100)

        fig.savefig(os.path.join(output_path,names[i]+'_PDF.png'),bbox_inches='tight')
        plt.clf()



def error(target_list,target_type,names,predctions,output_path):
    
    

    import os
    import numpy as np
    import pandas as pd
    from numba import njit

    @njit(cache=True,parallel=True)    
    def cal_func(target_list,predctions):
        
        fluc_predict=predctions-np.mean(predctions)
        fluc_target=target_list-np.mean(target_list)
        

        #Global average errors
        global_mean_err=(np.mean(predctions)-np.mean(target_list))/(np.mean(target_list))*100
        MSE_local_shear_stress=np.sqrt((np.mean((predctions-target_list)**2))/np.mean(target_list)**2)*100
        global_fluct_error=(np.std(fluc_predict)-np.std(fluc_target))/(np.std(fluc_target))*100
        MSE_local_fluc=np.sqrt((np.mean((fluc_predict-fluc_target)**2))/np.std(fluc_target)**2)*100

        

        #MAE_local=np.mean(np.abs(predctions[i][:,:,:]-target_list[i][:,:,:]))/np.mean(np.abs(target_list[i][:,:,:]))*100
        #MAE_local_no_mean=(np.abs(predctions[i][:,:,:]-target_list[i][:,:,:]))/np.mean(np.abs(target_list[i][:,:,:]))*100
        #MAE_fluct_no_mean=(np.abs(fluc_predict-fluc_target))/np.mean(np.abs(fluc_target))*100
        

        

        #Local erros for PDF's and boxplots etc.
        MSE_local_no_mean=np.sqrt(((predctions-target_list)**2)/np.mean(target_list)**2)*100
        #MSE_local_fluc_PDF=np.sqrt(((fluc_predict-fluc_target)**2)/(np.std(fluc_target))**2)*100
        
        return MSE_local_no_mean,global_mean_err,MSE_local_shear_stress,global_fluct_error,MSE_local_fluc


    if not os.path.exists(output_path):
        os.makedirs(output_path)

 
    
    if target_type=="stress":
        error=pd.DataFrame(columns=['Global Mean Error','Root mean sq. error of local shear stress','Global fluctuations error','Root mean sq. error of local fluctuations'])
    elif target_type=="flux":
        error=pd.DataFrame(columns=['Global Mean Error','Root mean sq. error of local heat flux','Global fluctuations error','Root mean sq. error of local fluctuations'])
    
    error_fluc_list=[]
    
    


    
    for i in range(3):
        error_fluct=pd.DataFrame()
        
        MSE_local_no_mean,global_mean_err,MSE_local_shear_stress,global_fluct_error,MSE_local_fluc=cal_func(target_list[i],predctions[i])


        if target_type=="stress":
            error_fluct['Root sq. error of local shear stress']=MSE_local_no_mean.flatten()
            error=error.append({'Global Mean Error':global_mean_err,'Root mean sq. error of local shear stress':MSE_local_shear_stress,'Global fluctuations error':global_fluct_error,'Root mean sq. error of local fluctuations':MSE_local_fluc},ignore_index=True)
        elif target_type=="flux":
            error_fluct['Root sq. error of local heat flux']=MSE_local_no_mean.flatten()
            error=error.append({'Global Mean Error':global_mean_err,'Root mean sq. error of local heat flux':MSE_local_shear_stress,'Global fluctuations error':global_fluct_error,'Root mean sq. error of local fluctuations':MSE_local_fluc},ignore_index=True)
        
        #error_fluct['Root sq. error of local fluctuations']=MSE_local_fluc_PDF.flatten()
        #error_fluct['MAE local']=MAE_local_no_mean.flatten()
        #error_fluct['MAE fluct']=MAE_fluct_no_mean.flatten()

        

        error_fluct.to_parquet(os.path.join(output_path,'Error_fluct_'+names[i]+'.parquet'))
        error_fluc_list.append(error_fluct)
        

    
    error.index=names

    error.to_csv(os.path.join(output_path,'Mean_error.csv'))

    return error_fluc_list, error



def heatmap_quarter_test(predction,target_var,output_path,target):
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    name='test'
    cm = 1/2.54  # centimeters in inches




    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Q_avg=0.665





    if target[0]=='tau_wall':
        target_var=target_var[1,:,:]/u_tau**2
        predction=predction[1,:,:]/u_tau**2        

        #cut the data to 1/4
        target_var=target_var[:128,:128]
        predction=predction[:128,:128]
            
    elif target[0][-5:]=='_flux':
        fric_temp=Q_avg/u_tau
        target_var=target_var[1,:,:]/Q_avg
        predction=predction[1,:,:]/Q_avg  

        #cut the data to 1/4
        target_var=target_var[:128,:128]
        predction=predction[:128,:128]

        #Need to find the average surface heat flux Q_w
        #Friction temp = Q_w/(u_tau)
        #q^+= q/(Friction temp)



    #Find highest and lowest value to scale plot to
    max_tot=np.max([np.max(target_var),np.max(predction)])
    min_tot=np.min([np.min(target_var),np.min(predction)])



    



    #max length in plus units
    x_range=12/2
    z_range=6/2

    gridpoints_x=int(255/2)+1
    gridponts_z=int(255/2)+1


    x_plus_max=x_range*u_tau/nu
    z_plus_max=z_range*u_tau/nu


    x_plus_max=np.round(x_plus_max).astype(int)
    z_plus_max=np.round(z_plus_max).astype(int)

    axis_range_x=np.array([0,470,950,1420,1900,2370])
    axis_range_z=np.array([0,295,590,890,1185])


    placement_x=axis_range_x*nu/u_tau
    placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


    placement_z=axis_range_z*nu/u_tau
    placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

    
    fig, axs=plt.subplots(2,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)

    #Target
    pcm=axs[0].imshow(np.transpose(target_var),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5,interpolation='bicubic')
    axs[0].set_title(name.capitalize(),weight="bold")
    axs[0].set_ylabel(r'$z^+$')
    
    #prediction
    axs[1].imshow(np.transpose(predction),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5,interpolation='bicubic')
    axs[1].set_xlabel(r'$x^+$')
    axs[1].set_ylabel(r'$z^+$')

    axs[1].set_xticks(placement_x)
    axs[1].set_xticklabels(axis_range_x,rotation=45)
    axs[0].set_yticks(placement_z)
    axs[0].set_yticklabels(axis_range_z)
    axs[1].set_yticks(placement_z)
    axs[1].set_yticklabels(axis_range_z)

        
    #Setting labels and stuff
    axs[0].text(-0.42, 0.20, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0].transAxes,rotation=90,weight="bold")

    axs[1].text(-0.42, 0.00, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=-0.31,hspace=0.25)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=20,shrink=0.7,location="bottom",pad=0.22)
    cbar.formatter.set_powerlimits((0, 0))


    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$',rotation=0)
    elif target[0]=='pr1_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=1$',rotation=0)
    elif target[0]=='pr0.2_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.2$',rotation=0)
    elif target[0]=='pr0.025_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max(target_var-predction)
    min_diff=np.min(target_var-predction)



    fig2, ax=plt.subplots(1,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)

    pcm=ax.imshow(np.transpose(target_var-predction),cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5,interpolation='bicubic')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_title("difference".capitalize(),weight="bold")
    ax.set_xticks(placement_x,)
    ax.set_xticklabels(axis_range_x,rotation=45)
    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x,rotation=45)

    ax.set_yticks(placement_z)
    ax.set_yticklabels(axis_range_z)
    cbar=fig2.colorbar(pcm,ax=ax,aspect=20,shrink=0.9,orientation="horizontal",pad=0.23)
    cbar.formatter.set_powerlimits((0, 0))

    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$',rotation=0)
    elif target[0]=='pr1_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=1$',rotation=0)
    elif target[0]=='pr0.2_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.2$',rotation=0)
    elif target[0]=='pr0.025_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',format='pdf')






def heatmap_quarter(predctions,target_list,output_path,target):
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    names=['training',"validation",'test']
    cm = 1/2.54  # centimeters in inches




    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Q_avg=0.665





    if target[0]=='tau_wall':
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/u_tau**2
            predctions[i]=predctions[i][1,:,:]/u_tau**2        

            #cut the data to 1/4
            target_list[i]=target_list[i][:128,:128]
            predctions[i]=predctions[i][:128,:128]
            
    elif target[0][-5:]=='_flux':
        fric_temp=Q_avg/u_tau
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/Q_avg
            predctions[i]=predctions[i][1,:,:]/Q_avg  

            #cut the data to 1/4
            target_list[i]=target_list[i][:128,:128]
            predctions[i]=predctions[i][:128,:128]

        #Need to find the average surface heat flux Q_w
        #Friction temp = Q_w/(u_tau)
        #q^+= q/(Friction temp)



    #Find highest and lowest value to scale plot to
    max_tot=0
    min_tot=1000
    for i in range(3):
        max_inter=np.max([np.max(target_list[i]),np.max(predctions[i])])
        min_inter=np.min([np.min(target_list[i]),np.min(predctions[i])])
        
        
        if max_inter>max_tot:
            max_tot=max_inter
        if min_inter<min_tot:
            min_tot=min_inter


    fig, axs=plt.subplots(2,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)



    #max length in plus units
    x_range=12/2
    z_range=6/2

    gridpoints_x=int(255/2)+1
    gridponts_z=int(255/2)+1


    x_plus_max=x_range*u_tau/nu
    z_plus_max=z_range*u_tau/nu


    x_plus_max=np.round(x_plus_max).astype(int)
    z_plus_max=np.round(z_plus_max).astype(int)

    axis_range_x=np.array([0,470,950,1420,1900,2370])
    axis_range_z=np.array([0,295,590,890,1185])


    placement_x=axis_range_x*nu/u_tau
    placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)


    placement_z=axis_range_z*nu/u_tau
    placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

    for i in range(3):  

        #Target
        pcm=axs[0,i].imshow(np.transpose(target_list[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[0,i].set_title(names[i].capitalize(),weight="bold")
        axs[0,0].set_ylabel(r'$z^+$')
        
        #prediction
        axs[1,i].imshow(np.transpose(predctions[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[1,i].set_xlabel(r'$x^+$')
        axs[1,0].set_ylabel(r'$z^+$')

        axs[1,0].set_xticks(placement_x)
        axs[1,0].set_xticklabels(axis_range_x,rotation=45)
        axs[1,1].set_xticks(placement_x)
        axs[1,1].set_xticklabels(axis_range_x,rotation=45)
        axs[1,2].set_xticks(placement_x)
        axs[1,2].set_xticklabels(axis_range_x,rotation=45)
        axs[0,0].set_yticks(placement_z)
        axs[0,0].set_yticklabels(axis_range_z)
        axs[1,0].set_yticks(placement_z)
        axs[1,0].set_yticklabels(axis_range_z)

        
    #Setting labels and stuff
    axs[0,0].text(-0.45, 0.20, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0,0].transAxes,rotation=90,weight="bold")

    axs[1,0].text(-0.45, 0.00, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1,0].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=-0.31,hspace=0.25)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=30,shrink=0.55,location="bottom",pad=0.24)
    cbar.formatter.set_powerlimits((0, 0))


    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$',rotation=0)
    elif target[0]=='pr1_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=1$',rotation=0)
    elif target[0]=='pr0.2_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.2$',rotation=0)
    elif target[0]=='pr0.025_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max([np.max(target_list[0]-predctions[0]),np.max(target_list[1]-predctions[1]),np.max(target_list[2]-predctions[2])])
    min_diff=np.min([np.min(target_list[0]-predctions[0]),np.min(target_list[1]-predctions[1]),np.min(target_list[2]-predctions[2])])

    fig2, axs=plt.subplots(1,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)
    for i in range(3):
        pcm=axs[i].imshow(np.transpose(target_list[i]-predctions[i]),cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5)
        axs[i].set_xlabel(r'$x^+$')
        axs[0].set_ylabel(r'$z^+$')
        axs[i].set_title(names[i].capitalize(),weight="bold")
        axs[0].set_xticks(placement_x,)
        axs[0].set_xticklabels(axis_range_x,rotation=45)
        axs[1].set_xticks(placement_x)
        axs[1].set_xticklabels(axis_range_x,rotation=45)
        axs[2].set_xticks(placement_x)
        axs[2].set_xticklabels(axis_range_x,rotation=45)

    axs[0].set_yticks(placement_z)
    axs[0].set_yticklabels(axis_range_z)
    fig2.subplots_adjust(wspace=0.13,hspace=0.05)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=30,shrink=0.55,location="bottom",pad=0.20)
    cbar.formatter.set_powerlimits((0, 0))

    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$',rotation=0)
    elif target[0]=='pr1_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=1$',rotation=0)
    elif target[0]=='pr0.2_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.2$',rotation=0)
    elif target[0]=='pr0.025_flux':
        cbar.ax.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.025$',rotation=0)
    else: 
        raise Exception('target name is not defined')


    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',format='pdf')

    return None

def heatmaps(target_list,names,predctions,output_path,model_path,target):
    """makes heatmaps of the Train validation and test data for target and prediction. Also plots the difference. Save to the output folder

    Args:
        target_list (list): list of arrays of the target
        names (list): list of names for the target_list
        predctions (list): list of array of the prediction
        output_path (Path): Path to the output folder
        model_path (Path): Path to the saved model

    Raises:
        Exception: if the target has no defined plot name
        Exception: Same as above

    Returns:
        None: 
    """
    from DataHandling import utility
    from DataHandling.features import slices
    from tensorflow import keras
    import numpy as np
    import shutil
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = 1/2.54  # centimeters in inches


 

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #change the scale to plus units
    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Q_avg=0.665
    
    
    if target[0]=='tau_wall':
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/u_tau
            predctions[i]=predctions[i][1,:,:]/u_tau        

            #cut the data to 1/4
            target_list[i]=target_list[i][128,128]
            predctions[i]=predctions[i][128,128]
            

    
    elif target[0][-5:] =='_flux':
        fric_temp=Q_avg/u_tau
        for i in range(len(target_list)):
            target_list[i]=target_list[i][1,:,:]/Q_avg
            predctions[i]=predctions[i][1,:,:]/Q_avg  

            #cut the data to 1/4
            target_list[i]=target_list[i][128,128]
            predctions[i]=predctions[i][128,128]

        #Need to find the average surface heat flux Q_w
        #Friction temp = Q_w/(u_tau)
        #q^+= q/(Friction temp)


    #Find highest and lowest value to scale plot to
    max_tot=0
    min_tot=1000
    for i in range(3):
        max_inter=np.max([np.max(target_list[i]),np.max(predctions[i])])
        min_inter=np.min([np.min(target_list[i]),np.min(predctions[i])])
        
        
        if max_inter>max_tot:
            max_tot=max_inter
        if min_inter<min_tot:
            min_tot=min_inter


    fig, axs=plt.subplots(2,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=150)


    #TODO lavet det her om så akserne passer på den 1/4 cut jeg har lavet
    
    #max length in plus units
    x_plus_max=12*u_tau/nu
    z_plus_max=6*u_tau/nu


    #To display the correct axis on the plot
    


    axis_range_x=np.linspace(0,255,7)
    x_axis_range=(axis_range_x-0)/(255-0)*(12-0)+0
    x_axis_range=np.round(x_axis_range/u_tau).astype(int)
    
    axis_range_z=np.linspace(0,255,4)
    z_axis_range=(axis_range_z-0)/(255-0)*(6-0)+0
    z_axis_range=np.flip(z_axis_range)
    z_axis_range=np.round(z_axis_range*u_tau/nu).astype(int)
    for i in range(3):  

        #Target
        pcm=axs[0,i].imshow(np.transpose(target_list[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[0,i].set_title(names[i].capitalize(),weight="bold")
        axs[0,0].set_ylabel(r'$z^+$')
        
        #prediction
        axs[1,i].imshow(np.transpose(predctions[i]),cmap='viridis',vmin=min_tot,vmax=max_tot,aspect=0.5)
        axs[1,i].set_xlabel(r'$x^+$')
        axs[1,0].set_ylabel(r'$z^+$')

        axs[1,i].set_xticks(axis_range_x)
        axs[1,i].set_xticklabels(x_axis_range)
        axs[0,0].set_yticks(axis_range_z)
        axs[0,0].set_yticklabels(z_axis_range)
        axs[1,0].set_yticks(axis_range_z)
        axs[1,0].set_yticklabels(z_axis_range)

        
    #Setting labels and stuff
    axs[0,0].text(-0.23, 0.30, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0,0].transAxes,rotation=90,weight="bold")

    axs[1,0].text(-0.23, 0.20, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1,0].transAxes,rotation=90,weight="bold")

    fig.subplots_adjust(wspace=0.09,hspace=0.15)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=20,shrink=0.5,location="bottom")
    cbar.formatter.set_powerlimits((0, 0))


    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$q_w^+,\quad Pr=0.71$',rotation=0)
    else: 
        raise Exception('target name is not defined')

    fig.savefig(os.path.join(output_path,'target_prediction.pdf'),bbox_inches='tight',format='pdf')


    max_diff=np.max([np.max(target_list[0]-predctions[0]),np.max(target_list[1]-predctions[1]),np.max(target_list[2]-predctions[2])])
    min_diff=np.min([np.min(target_list[0]-predctions[0]),np.min(target_list[1]-predctions[1]),np.min(target_list[2]-predctions[2])])

    fig2, axs=plt.subplots(1,3,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=150)
    for i in range(3):
        pcm=axs[i].imshow(target_list[i]-predctions[i],cmap="Spectral",vmin=min_diff,vmax=max_diff,aspect=0.5)
        axs[i].set_xlabel(r'$x^+$')
        axs[0].set_ylabel(r'$z^+$')
        axs[i].set_title(names[i].capitalize(),weight="bold")
        axs[i].set_xticks(axis_range_x)
        axs[i].set_xticklabels(x_axis_range)

    axs[0].set_yticks(axis_range_z)
    axs[0].set_yticklabels(z_axis_range)
    fig2.subplots_adjust(wspace=0.09,hspace=0.05)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=20,shrink=0.5,location="bottom")
    cbar.formatter.set_powerlimits((0, 0))

    if target[0]=='tau_wall':
        cbar.ax.set_xlabel(r'$\tau_{w}^{+} $',rotation=0)
    elif target[0]=='pr0.71_flux':
        cbar.ax.set_xlabel(r'$q_w^+,\quad Pr=0.71$',rotation=0)
    else: 
        raise Exception('target name is not defined')



    fig2.savefig(os.path.join(output_path,'difference.pdf'),bbox_inches='tight',format='pdf')
    
    return None





def stat_plots(mean_dataset_loc,batches):
    from DataHandling.features.stats import get_valdata
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    
    mean = xr.open_mfdataset(mean_dataset_loc, parallel=True)
    mean = mean.persist()
    mean = mean.groupby_bins("time", batches).mean()

    #Validation data
    val_u = get_valdata('u')

    linerRegion = np.linspace(0, 9)
    logRegion = 1 / 0.4 * np.log(np.linspace(20, 180)) + 5
    figScale = 2
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12 * figScale, 6 * figScale))
    ax1.plot('y+', 'u_plusmean', 'ok', data=val_u, label='DNS validation data')
    colorList = ['*b', '*c', '*y', '*g', '.b', '.c', '.y', '.g', 'vb', 'vc', 'vy', 'vg', '<b', '<c', '<y', '<g'] * 30
    # Plotting the batches in mean for U
    for i in range(len(mean.time_bins)):
        ax1.plot(mean.y_plus, mean.u_plusmean.isel(time_bins=i), colorList[i], label='DNS batch ' + str(i))

    ax1.plot(linerRegion, linerRegion, 'r', label='Linear Region')
    ax1.plot(np.linspace(20, 180), logRegion, 'm', linewidth=5, label='Log Region')

    ax1.set_title('Normalized mean values')
    ax1.set_xscale('log')
    ax1.set_xlabel('$y^{+}$')
    ax1.set_ylabel('$<u^{+}>$')
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0, 20)
    ax1.minorticks_on()
    ax1.grid(True, which="both", linestyle='--')
    ax1.legend(prop={"size": 17})

    # Now for <u_rms>

    ax2.plot('y+', 'u_plusRMS', 'ok', data=val_u, label='DNS validation data')
    ax2.set_title('Normalized RMS of fluctuations')
    for i in range(len(mean.time_bins)):
        ax2.plot(mean.y_plus, mean.u_plusRMS.isel(time_bins=i), colorList[i], label='DNS batch ' + str(i))

    ax2.set_xscale('log')
    ax2.set_xlabel('$y^{+}$')
    ax2.set_ylabel("$u^{+}_{RMS}$")
    ax2.set_xlim(1, 300)
    ax2.minorticks_on()
    ax2.grid(True, which="both", linestyle='--')
    ax2.legend(prop={"size": 17})
    plt.tight_layout()
    plt.savefig("/home/au569913/DataHandling/reports/figures/u_val.pdf", bbox_inches='tight')

    a = get_valdata('pr1')
    b = get_valdata('pr71')
    c = get_valdata('pr0025')

    val_pr = [a, b, c]
    val_pr = val_pr[0].join(val_pr[1:])

    linerRegion = np.linspace(0, 9)
    logRegion = 1 / 0.43 * np.log(np.linspace(20, 180)) + 3
    figScale = 2

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12 * figScale, 6 * figScale))
    ax1.plot('pr1_y+', 'pr1_plusmean', 'ok', data=val_pr, label='DNS validation data Pr=1')
    ax1.plot('pr0.71_y+', 'pr0.71_plusmean', 'or', data=val_pr, label='DNS validation data Pr=0.71')
    ax1.plot('pr0.025_y+', 'pr0.025_plusmean', 'om', data=val_pr, label='DNS validation data Pr=0.025')
    colorList = ['*b', '*c', '*y', '*g', '.b', '.c', '.y', '.g', 'vb', 'vc', 'vy', 'vg', '<b', '<c', '<y', '<g'] * 30

    # Plotting the batches in mean for the different Pr
    pr_list = ['pr1', 'pr0.71', 'pr0.2', 'pr0.025']
    j = 0
    for i in range(len(mean.time_bins)):
        for Pr in pr_list:
            ax1.plot(mean.y_plus, mean[Pr + '_plusmean'].isel(time_bins=i), colorList[j * 4 + i],
                     label='DNS batch' + str(i) + 'Pr=' + Pr[2:])
            j = j + 1

    dualColor = ['k', 'r']
    j = 0
    for Pr in pr_list:
        ax1.plot(linerRegion, linerRegion * float(Pr[2:]), dualColor[i % 2],
                 label='Linear Region y+*Pr ' + 'Pr=' + Pr[2:])
        j = j + 1

    ax1.set_xscale('log')
    ax1.set_xlabel('$y^+$')
    ax1.set_ylabel(r'$<\theta^{+}>$')
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0, 20)
    ax1.grid(True, which="both", linestyle='--')
    ax1.legend(loc='best', prop={"size": 15})

    # Now for <Pr_rms>

    ax2.plot('pr1_y+', 'pr1_plusRMS', 'ok', data=val_pr, label='DNS validation data Pr=1')
    ax2.plot('pr0.71_y+', 'pr0.71_plusRMS', 'or', data=val_pr, label='DNS validation data Pr=0.71')
    ax2.plot('pr0.025_y+', 'pr0.025_plusRMS', 'om', data=val_pr, label='DNS validation data Pr=0.025')
    ax2.set_title('Normalized RMS of fluctuations')
    j = 0
    for i in range(len(mean.time_bins)):
        for Pr in pr_list:
            ax2.plot(mean.y_plus, mean[Pr + '_plusRMS'].isel(time_bins=i), colorList[j * 4 + i],
                     label='DNS batch' + str(i) + 'Pr=' + Pr[2:])
            j = j + 1

    ax2.set_xscale('log')
    ax2.set_xlabel('$y^+$')
    ax2.set_ylabel(r'$\theta ^{+}_{RMS}$')
    ax2.set_xlim(1, 300)
    ax2.set_ylim(0, 3)
    ax2.grid(True, which="both", linestyle='--')
    ax2.legend(loc='best', prop={"size": 15})

    plt.tight_layout()
    plt.savefig("/home/au569913/DataHandling/reports/figures/Pr_val.pdf", bbox_inches='tight')




#--- fgn --#

def dsfield(ds,domain,dim='y',save=True):
    """"2D u velocity slice 

    Args:
        ds (xarray)
        dim (str): dimension to slice
    """
    import matplotlib.pyplot as plt
    u_tau = 0.05
    nu = 0.0004545454545
    ds = ds.copy()
    ds.coords['x']=ds.coords['x']*(u_tau/nu)
    ds.coords['y'] = abs(ds.coords['y']-ds.coords['y'].max())*(u_tau/nu) #y_plus
    ds.coords['z'] = (ds.coords['z'] + (-ds.coords['z'][0]))*(u_tau/nu)
    #ds['u_vel']=ds['u_vel']/ds['u_vel'].max()
    print('done coords')
    if dim=='y':
        time = 4200 # ind=400
        y = 1.882 # ind = 10
        uin = ds.u_vel.isel(time=3000,y=10)
    elif dim=='z':
        time = 3003
        z = 0
        uin = ds.u_vel.isel(time=0,z=16)

    uin = uin/uin.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cbar_frac = 0.025 if dim == 'y' else 0.025
    levels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    uin.T.plot.contourf(ax=ax,levels=200,cmap='jet',vmin=0, cbar_kwargs={'fraction':cbar_frac, 'ticks':levels})
    print('done plot')
    #fig.axes[1].remove()
    fig.axes[1].set_ylabel(r'$u/u_{max}$')
    ax.set_aspect('equal')
    ax.set_title(r'$t_{{e}}={:.0f},{}^{{+}}={:.2f}$'.format(uin.coords['time'].values,dim,uin.coords[dim].values))
    if dim == 'y':
        ax.set_ylabel(r'${}^{{+}}$'.format('z'))
    elif dim == 'z':
        ax.set_ylabel(r'${}^{{+}}$'.format('y'))
    ax.set_xlabel(r'$x^{+}$')

    #Remove annoying white lines
    for c in ax.collections:
        c.set_edgecolor("face")

    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/{}_dsfield_{}.pdf".format(domain,dim,domain),bbox_inches='tight')


def uslice(predctions,target_list,name,domain,dim,save=True):
    """"2D u velocity slice plot of target/pred

    Args:
        predctions (list): list of the train,validation,test predictions
        target_list (list): list of the train,validation,test target
        names (list): list of the names of the data. Normally train,validaiton,test
        name (str): name of save
        ds (xrray)
        dim (str): dimension to slice
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import seaborn as sns
    import xarray as xr
    import numpy as np
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
    ds=ds.isel(y=slice(0, 32))
    #x, y = np.meshgrid(ds['x'].values, ds['y'].values)
    x = ds['x'].values
    y = ds['y'].values
    z = ds['z'].values

    u_tau = 0.05
    nu = 0.0004545454545
    
    x = x*(u_tau/nu)
    y = abs(y-y.max())*(u_tau/nu) #y_plus
    z = (z + (-z[0]))*(u_tau/nu)
    xtitle = r'$x^{+}$'
    if dim == 'y':
    #Create meshgrid
        x, y = np.meshgrid(x, y, indexing='xy')
    elif dim == 'z':
        x, y = np.meshgrid(x, z, indexing='xy')
    


    #plt.scatter(x, y,0.1)
    #segs1 = np.stack((x,y), axis=2)
    #segs2 = segs1.transpose(1,0,2)
    #plt.gca().add_collection(LineCollection(segs1))
    #plt.gca().add_collection(LineCollection(segs2))

    time = 100
    #z = ds.isel(time=200,z=16)['u_vel'].values.T
    if dim == 'y':
        z1 = target_list[time,:,:,16,0].T
        z2 = predctions[time,:,:,16,0].T
        
    if dim == 'z':
        z1 = target_list[time,:,16,:,0].T
        z2 = predctions[time,:,16,:,0].T
        
    if dim == 'POD':
        z1 = target_list
        z2 = predctions
        

    cm = 1/2.54  # centimeters in inches
    #name='test'

    fig, axs=plt.subplots(2,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)
    #Target
    pcm=axs[0].contourf(x,y,z1,levels=200,cmap='jet', vmin=min([z1.min(), z2.min()]), vmax=max([z1.max(), z2.max()]))
    axs[0].contourf(x,y,z1,levels=200,cmap='jet', vmin=min([z1.min(), z2.min()]), vmax=max([z1.max(), z2.max()])) #twice bc it apparently helps with removing white lines
    axs[0].set_title(name,weight="bold") #title
    axs[0].set_ylabel(r'${}^{{+}}$'.format(dim))

    #prediction
    axs[1].contourf(x,y,z2,levels=200,cmap='jet', vmin=min([z1.min(), z2.min()]), vmax=max([z1.max(), z2.max()]))
    axs[1].contourf(x,y,z2,levels=200,cmap='jet', vmin=min([z1.min(), z2.min()]), vmax=max([z1.max(), z2.max()])) #twice bc it apparently helps with removing white lines
    axs[1].set_xlabel(r'$x^{+}$')
    axs[1].set_ylabel(r'${}^{{+}}$'.format(dim))

    #Setting labels and stuff
    axs[0].text(-0.3, 0.20, 'Target',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[0].transAxes,rotation=90,weight="bold")
    axs[1].text(-0.3, 0.00, 'Prediction',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1].transAxes,rotation=90,weight="bold")
    fig.subplots_adjust(wspace=-0.31,hspace=0.25)
    cbar=fig.colorbar(pcm,ax=axs[:],aspect=20,shrink=1.0,location="bottom",pad=0.2)
    cbar.formatter.set_powerlimits((0, 0))
    ticks = np.linspace(int(min([z1.min() ,z2.min()])), int(max([z1.max() ,z2.max()])), 5, endpoint=True)
    #cbar.set_ticks([-5 -2.5, 0.0, 2.5, 5]), cbar.set_ticklabels([-5 -2.5, 0.0, 2.5, 5])
    print(ticks)
    cbar.set_ticks(ticks), cbar.set_ticklabels(ticks)
    #cbar.set_format('%0.1f')
    cbar.ax.set_xlabel(r"$u^{'+}$",rotation=0)
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/uslice_{}_{}.pdf".format(domain,name,dim),bbox_inches='tight')
    plt.show()

def isocon(data,ds,name,domain,type,save=True):
    """"3D isocontour

    Args:
        ds (xarray): Xarray dataset for coordinates
        data (nparray): nparray
        name (str): String of name to save and put in title
        domain (str): Domain e.g. nakamura/blonigan
    """
    import plotly.graph_objects as go
    from scipy import interpolate
    import numpy as np
    import plotly.io as pio
    #pio.kaleido.scope.default_format = "svg"
    # Data coordinates
    x = ds['x'].values
    y = ds['y'].values
    z = ds['z'].values

    u_tau = 0.05
    nu = 0.0004545454545
    
    x = x*(u_tau/nu)
    y = abs(y-y.max())*(u_tau/nu) #y_plus
    z = (z + (-z[0]))*(u_tau/nu)
    xtitle = r'$x^{+}$'
    #Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    #Convert to 32768,3
    points = np.array((X.flatten(), Y.flatten(), Z.flatten())).T

    # Data values @ above coords
    values = data.flatten().T
    
    # Grid to interpolate to
    #Xnew, Ynew, Znew = np.mgrid[x.min():x.max():len(x)*2j, y.min():y.max():len(y)*2j, z.min():z.max():len(z)*2j]
    # New data values on interp grid
    #newdata = interpolate.griddata(points, values, (Xnew,Ynew,Znew) )
    print('Done')
    if type == 'Qcrit':
        im = 0.01
        sc = 1
        showscale = False
        colorscale ='amp'
    if type == 'uvel':
        im = -0.01
        sc = 2
        showscale = True
        colorscale = 'jet'
    # make the plot
    fig = go.Figure(data=[
            go.Isosurface(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),
                        value=values.flatten(),
                        opacity=0.95, #0.9
                        #surface_fill=0.6,
                        isomin=im, # changes based on qcrit or uvel
                        isomax=0.01,
                        surface_count=sc,
                        lighting=dict(ambient=0.7), #0.7
                        colorscale=colorscale,
                        showscale=showscale,
                        caps=dict(x_show=False, y_show=False)
                        ),
                    ])
    if type == 'uvel':
        fig = go.Figure(data=[
                go.Isosurface(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),
                            value=values.flatten(),
                            opacity=0.9,
                            #surface_fill=0.6,
                            isomin=0.01, # changes based on qcrit or uvel
                            isomax=0.01,
                            surface_count=1,
                            lighting=dict(ambient=0.7),
                            colorscale=colorscale,
                            showscale=False,
                            caps=dict(x_show=False, y_show=False)
                            ),
                go.Isosurface(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),
                            value=values.flatten(),
                            opacity=0.9,
                            #surface_fill=0.6,
                            isomin=-0.01, # changes based on qcrit or uvel
                            isomax=-0.01,
                            surface_count=1,
                            lighting=dict(ambient=0.7), 
                            colorscale='Blues',
                            showscale=False,
                            caps=dict(x_show=False, y_show=False),
                            ),
                        ])
    
    
    scene = dict(
                    xaxis_title=r"x+",
                    yaxis_title=r'y+',
                    zaxis_title=r'z+'
    )
    # Default parameters which are used when `layout.scene.camera` is not provided
    camera = dict(
        up=dict(x=0.1, y=0.8, z=0.1), #x=0.1, y=0.8, z=0.1
        center=dict(x=-0.3, y=-0.5, z=-0.4), #x=0, y=-0.5, z=0
        eye=dict(x=1.2, y=0.9, z=1.6) #x=1.3, y=1, z=1.8
    )
    #fig.update_traces(colorbar_tickvals=[-0.01,0.01])
    
    if isinstance(name, str):
        title = name
        title = None
    else:
        title=r'$t_{{e}}={:.0f}$'.format(name)
        name = 't={:.0f}'.format(name)
    
    fig.update_layout(scene=scene,scene_camera=camera, title_text=title, title_x=0.5)
    fig.update_traces(colorbar_len=0.5, colorbar_thickness=5,colorbar_nticks=3)
    fig.update_layout(
        autosize=False,
        width=1030, #515
        height=600, #200
        margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=30,
        pad = 0
    )
)
    if save == True:
        fig.write_image("/home/au569913/DataHandling/reports/{}/{}_isocon_{}.pdf".format(domain,name,domain),format='pdf')
    fig.show(renderer="svg")
    #fig.show()
    
#%%
def rmsplot(model,target,pred1,pred2,pred3,ds,domain,save=True,scae=[]):
    from DataHandling.features import preprocess
    from DataHandling import postprocess
    import matplotlib.pyplot as plt
    rms_tar = preprocess.rms(target)
    rms_pred1 = preprocess.rms(pred1)
    rms_pred2 = preprocess.rms(pred2)
    rms_pred3 = preprocess.rms(pred3) 

    u_tau = 0.05
    nu = 0.0004545454545
    y = ds.coords['y'].values
    y = abs(y-2)*(u_tau/nu) #y_plus

    if model == 'POD':
        rms_tar = rms_tar/u_tau
        rms_pred1 = rms_pred1/u_tau
        rms_pred2 = rms_pred2/u_tau
        rms_pred3 = rms_pred3/u_tau 

    name = model
    
    labels = ['DNS',r'$r=1536$',r'$r=192$',r'$r=24$']

    if domain == '1pi':
        labels = ['DNS',r'$r=3072$',r'$r=384$',r'$r=48$']
    if name == 'SCAE':
        r0 = postprocess.mediancomp(scae[0])
        r1 = postprocess.mediancomp(scae[1])
        r2 = postprocess.mediancomp(scae[2])
        labels = ['DNS',r'$r={}$'.format(r0),r'$r={}$'.format(r1),r'$r={}$'.format(r2)]
    cm = 1/2.54  # centimeters in inches
    #fig, axs=plt.subplots(2,figsize=([7*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)
    fig, axs=plt.subplots(3,figsize=([7*cm,15*cm]),sharex=True,sharey=False,constrained_layout=True,dpi=1000)
    # u
    for i in range(0,3):
        axs[i].plot(y,rms_tar[:,i],lw=2)
        axs[i].plot(y,rms_pred1[:,i],lw=0.7)
        axs[i].plot(y,rms_pred2[:,i],lw=0.7)
        axs[i].plot(y,rms_pred3[:,i],lw=0.7)
        axs[i].legend(labels,prop={'size': 6})
        axs[i].grid(True)
    #name.capitalize()
    axs[0].set_title(name,weight="bold")
    axs[0].set_ylabel(r"$u_{\mathrm{rms}}^{'+}$")
    axs[0].grid(True)
    # v
    
    
    axs[1].set_ylabel(r"$v_{\mathrm{rms}}^{'+}$")
    axs[1].grid(True)
    # w


    axs[2].grid(True)
    axs[2].set_ylabel(r"$w_{\mathrm{rms}}^{'+}$")
    axs[2].set_xlabel(r'$y^{+}$')
    

    #Setting labels and stuff
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/{}_rms_{}.pdf".format(domain,name,domain),bbox_inches='tight')
    #plt.show()
#%%
def KE_plot(KE,domain,fluc=False,KE_pred=False,vlines=False,save=True):
    """"Plot Kinetic energy

    Args:
        KE (xarray): Xarray dataset for coordinates
        KE_pred (bool): Include scattered predictions
    """

    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    cm = 1/2.54  # centimeters in inches
    u_tau = 0.05

    fig, axs=plt.subplots(1,figsize=([17*cm,7*cm]),sharex=True,sharey=False,constrained_layout=True,dpi=1000)
    KE.plot(ax = axs, color='k',lw=0.5)
    axs.set_xlabel(r'$t_{e}$')
    if fluc == True:
        axs.set_ylabel(r'$TKE^{+}$')
    else:
        axs.set_ylabel(r'$KE^{+}$')
    axs.grid(True)
    
    if KE_pred != False:
        if vlines == 'color':
            #maxKE=max(KE).values
            time_to_plot = KE.coords['time'][KE_pred[0]]
            plt.vlines(x = time_to_plot, ymin = min(KE), ymax = max(KE),
             colors = 'red', label = 'Badly recreated',lw = 0.5)
            plt.text(time_to_plot[3]+75,max(KE)-0.025*max(KE),r'$\it{(c)}$',rotation=0)
            time_to_plot = KE.coords['time'][KE_pred[1]]
            plt.vlines(x = time_to_plot, ymin = min(KE), ymax = max(KE),
             colors = 'blue', label = 'intermediate recreated',lw = 0.5)
            plt.text(time_to_plot[4]+75,max(KE)-0.025*max(KE),r'$\it{(b)}$',rotation=0)
            time_to_plot = KE.coords['time'][KE_pred[2]]
            plt.vlines(x = time_to_plot, ymin = min(KE), ymax = max(KE),
             colors = 'green', label = 'Well recreated',lw = 0.5)
            plt.text(time_to_plot[0]+75,max(KE)-0.025*max(KE),r'$\it{(a)}$',rotation=0)
        elif vlines == 'black':
            #time_to_plot = KE.sel(time=13056).coords['time']
            time_to_plot = 13056
            plt.vlines(x = time_to_plot, ymin = min(KE), ymax = max(KE),
             colors = 'grey', label = 'a',lw = 1, linestyles='dashed')
            plt.text(time_to_plot+100,max(KE)-0.025*max(KE),r'$\it{(a)}$',rotation=0)
            time_to_plot = 3846
            plt.vlines(x = time_to_plot, ymin = min(KE), ymax = max(KE),
             colors = 'grey', label = 'b',lw = 1, linestyles='dashed')
            plt.text(time_to_plot-700,max(KE)-0.025*max(KE),r'$\it{(b)}$',rotation=0)
            time_to_plot = 4269
            plt.vlines(x = time_to_plot, ymin = min(KE), ymax = max(KE),
             colors = 'grey', label = 'c',lw = 1, linestyles='dashed')
            plt.text(time_to_plot+100,max(KE)-0.025*max(KE),r'$\it{(c)}$',rotation=0)


        else:
            test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")
            test_time = KE.coords['time'][test_ind]
    
            train_ind =np.load("/home/au569913/DataHandling/data/interim/train_ind.npy")
            train_time = KE.coords['time'][train_ind]

            plt.scatter(test_time,KE_pred*(u_tau**2),marker='.')
    

    if save == True:
        if vlines == 'color':
            plt.savefig("/home/au569913/DataHandling/reports/{}/KE_vlines_{}.pdf".format(domain,domain),bbox_inches='tight')
        elif vlines == 'black':
            plt.savefig("/home/au569913/DataHandling/reports/{}/KE_blacklines_{}.pdf".format(domain,domain),bbox_inches='tight')
        else:
            if fluc == False:
                plt.savefig("/home/au569913/DataHandling/reports/{}/KE_{}.pdf".format(domain,domain),bbox_inches='tight')
            elif fluc == True:
                plt.savefig("/home/au569913/DataHandling/reports/{}/TKE_{}.pdf".format(domain,domain),bbox_inches='tight')

#%%
def KE_arrangeplot(KE_total, KE_pred_total, KE_c3, KE_scae,domain,showscae=True,save=True):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from DataHandling import postprocess
    u_tau = 0.05

    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,figsize=([17*cm,7*cm]),sharex=True,sharey=False,constrained_layout=True,dpi=1000)

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

    #arr1inds = KE_total.isel(time=test_ind).values.argsort() # pick out indexes of sorted ds
    #plt.plot(np.arange(0,499,1),KE_total.isel(time=test_ind).values[arr1inds[::-1]],color='k')
    arr1inds = KE_total.values.argsort() # pick out indexes of sorted ds
    plt.plot(np.arange(0,499,1),KE_total.values[arr1inds[::-1]],lw=2)
    plt.scatter(np.arange(0,499,1),KE_c3[arr1inds[::-1]]/(u_tau**2),marker='.',s=8,color='C1')
    plt.scatter(np.arange(0,499,1),KE_pred_total[arr1inds[::-1]],marker='.',s=8,color='C2')
    if showscae == True:
        plt.scatter(np.arange(0,499,1),KE_scae[arr1inds[::-1]],marker='.',s=8,color='C3')
    plt.ylabel(r'$TKE^{+}$')
    plt.legend(['DNS','POD','CNNAE','SCAE'])
    #axs.grid(True)
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/KE_arranged_{}.pdf".format(domain, domain),bbox_inches='tight')

def KE_arrange5(KE_total, KE_pred_total, KE_c3,domain,KE_scae,cut='high',showscae=False,save=True):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from DataHandling import postprocess
    u_tau = 0.05

    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,figsize=([17*cm,7*cm]),sharex=True,sharey=False,constrained_layout=True,dpi=1000)

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

    #arr1inds = KE_total.isel(time=test_ind).values.argsort() # pick out indexes of sorted ds
    #plt.plot(np.arange(0,499,1),KE_total.isel(time=test_ind).values[arr1inds[::-1]],color='k')
    arr1inds = KE_total.values.argsort() # pick out indexes of sorted ds
    if cut == 'high':
        arr1inds=arr1inds[-25:]
    elif cut =='low':
        arr1inds=arr1inds[0:25]
    x = np.arange(0,25,1)
    plt.scatter(x,KE_total.values[arr1inds[::-1]],marker='.',s=50)
    plt.scatter(x,KE_c3[arr1inds[::-1]]/(u_tau**2),marker='.',s=20)
    plt.scatter(x,KE_pred_total[arr1inds[::-1]],marker='.',s=20)
    if showscae == True: 
        plt.scatter(x,KE_scae[arr1inds[::-1]],marker='.',s=20)
    plt.plot(x,KE_total.values[arr1inds[::-1]])
    plt.plot(x,KE_c3[arr1inds[::-1]]/(u_tau**2))
    plt.plot(x,KE_pred_total[arr1inds[::-1]])
    if showscae == True: 
        plt.plot(x,KE_scae[arr1inds[::-1]])
    plt.ylabel(r'$TKE^{+}$')
    plt.xticks(x)
    plt.legend(['DNS','POD','CNNAE','SCAE'])
    axs.grid(axis='y')
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/KE5_{}_{}.pdf".format(domain,cut,domain),bbox_inches='tight')
    abs_error = [np.mean(KE_total.values[arr1inds[::-1]]-KE_c3[arr1inds[::-1]]/(u_tau**2)), np.mean(KE_total.values[arr1inds[::-1]]-KE_pred_total[arr1inds[::-1]]),
    np.mean(KE_total.values[arr1inds[::-1]]-KE_scae[arr1inds[::-1]])]
    rel_error =  [x / np.mean(KE_total.values[arr1inds[::-1]]) for x in abs_error]
    return abs_error, rel_error

def scaemode(comp,name,domain,dim,save=True):
    """"2D slice plot 4 scae mode

    Args:
        comp (array): compressed space of network shape: (499,32,32,32,12)
        names (list): list of the names of the data. Normally train,validaiton,test
        name (str): name of save
        ds (xrray)
        dim (str): dimension to slice
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib
    import seaborn as sns
    import xarray as xr
    import numpy as np
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
    ds=ds.isel(y=slice(0, 32))
    #x, y = np.meshgrid(ds['x'].values, ds['y'].values)
    x = ds['x'].values
    y = ds['y'].values
    z = ds['z'].values

    u_tau = 0.05
    nu = 0.0004545454545
    
    x = x*(u_tau/nu)
    y = abs(y-y.max())*(u_tau/nu) #y_plus
    z = (z + (-z[0]))*(u_tau/nu)
    xtitle = r'$x^{+}$'
    if dim == 'y':
    #Create meshgrid
        x, y = np.meshgrid(x, y, indexing='xy')
    elif dim == 'z':
        x, y = np.meshgrid(x, z, indexing='xy')

    modecontent = comp.sum(axis=0).sum(axis=0).sum(axis=0).sum(axis=0) #shape 12
    mostcontent = modecontent.argsort()[::-1] #nparray of index of modes with most content (nonzeros)

    time = 100
    #z = ds.isel(time=200,z=16)['u_vel'].values.T
    if dim == 'y':
        z1 = comp[time,:,:,16,mostcontent[0]].T
        z2 = comp[time,:,:,16,mostcontent[1]].T
        z3 = comp[time,:,:,16,mostcontent[2]].T
        z4 = comp[time,:,:,16,mostcontent[3]].T
        
    if dim == 'z':
        z1 = comp[time,:,16,:,mostcontent[0]].T
        z2 = comp[time,:,16,:,mostcontent[1]].T
        z3 = comp[time,:,16,:,mostcontent[2]].T
        z4 = comp[time,:,16,:,mostcontent[3]].T
        

    cm = 1/2.54  # centimeters in inches
    #name='test'

    fig, axs=plt.subplots(2,2,figsize=([12*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)
    vmin = min([z1.min(), z2.min(), z3.min(), z4.min()])
    vmax = max([z1.max(), z2.max(), z3.max(), z4.max()])
    max_ax = np.argmax([z1.max(), z2.max(), z3.max(), z4.max()]) #if time scale colorbar pcm to the axs containing highest z value
    print(max_ax)
    #Mode 1
    pcm=axs[0,0].contourf(x,y,z1,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    axs[0,0].set_title('Mode {}'.format(mostcontent[0]),weight="bold") #title
    axs[0,0].set_ylabel(r'${}^{{+}}$'.format(dim))

    #Mode 2
    axs[1,0].contourf(x,y,z2,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    axs[1,0].set_xlabel(r'$x^{+}$')
    axs[1,0].set_ylabel(r'${}^{{+}}$'.format(dim))
    axs[1,0].set_title('Mode {}'.format(mostcontent[1]),weight="bold") #title
    #Mode 3
    axs[0,1].contourf(x,y,z3,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    #axs[2].set_xlabel(r'$x^{+}$')
    #axs[2].set_ylabel(r'${}^{{+}}$'.format(dim))
    axs[0,1].set_title('Mode {}'.format(mostcontent[2]),weight="bold") #title

    #Mode 4
    axs[1,1].contourf(x,y,z4,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    axs[1,1].set_xlabel(r'$x^{+}$')
    #axs[2].set_ylabel(r'${}^{{+}}$'.format(dim))
    axs[1,1].set_title('Mode {}'.format(mostcontent[3]),weight="bold") #title

    #Decide which one to use for colorbar
    if max_ax == 0:
        pcm=axs[0,0].contourf(x,y,z1,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    elif max_ax == 1:
        pcm=axs[1,0].contourf(x,y,z2,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    elif max_ax == 2:
        pcm=axs[0,1].contourf(x,y,z3,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    elif max_ax == 3:
        pcm=axs[1,1].contourf(x,y,z4,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    
    pcm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin,vmax),cmap='jet')

    #Setting labels and stuff
    #axs[0,0].text(-0.35, 0.20, 'Mode 1',
    #        verticalalignment='bottom', horizontalalignment='right',
    #        transform=axs[0,0].transAxes,rotation=90,weight="bold")
    #axs[1,0].text(-0.35, 0.20, 'Mode 2',
    #        verticalalignment='bottom', horizontalalignment='right',
    #        transform=axs[1,0].transAxes,rotation=90,weight="bold")
    #axs[0,1].text(-0.08, 0.20, 'Mode 3',
    #        verticalalignment='bottom', horizontalalignment='right',
    #        transform=axs[0,1].transAxes,rotation=90,weight="bold")
    #axs[1,1].text(-0.08, 0.20, 'Mode 4',
    #        verticalalignment='bottom', horizontalalignment='right',
    #        transform=axs[1,1].transAxes,rotation=90,weight="bold")
    #Remove annoying white lines
    for ax in axs.reshape(-1): 
        for c in ax.collections:
            c.set_edgecolor("face")
    fig.subplots_adjust(hspace=0.35)
    ticks = np.linspace(vmin, vmax, 5, endpoint=True)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=30,shrink=0.7,location="bottom",pad=0.18,
    format="%.2e",spacing='uniform',ticks=ticks)
    #cbar=fig.colorbar(cax=axs[:],aspect=20,shrink=1.0,location="bottom",pad=0.2)
    #cbar.formatter.set_powerlimits((0, 0))
    ticks = np.linspace(vmin, vmax, 5, endpoint=True)
    #cbar.set_ticks([-5 -2.5, 0.0, 2.5, 5]), cbar.set_ticklabels([-5 -2.5, 0.0, 2.5, 5])
    print(ticks)
    #cbar.set_ticks(ticks), cbar.set_ticklabels(ticks)
    #cbar.ax.set_xlabel(r"$u^{'+}$",rotation=0)
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/scaemode_{}_{}.pdf".format(domain,name,dim),bbox_inches='tight')
    plt.show()

def errorplot(poderror,cnnaeerror,scaeerror,domain,scae,save=True):
    """Error plot
    Args:
    domain (str): The domain on which the POD has been carried out.
    scae (list): name of scae domains to determine latent space size
    Returns:

    """
    import numpy as np
    import matplotlib.pyplot as plt
    from DataHandling import postprocess
    
    name = ''
    modes = 400

    if domain == '1pi':
        ticks = [3072,384,48]
    else:
        ticks = [1536,192,24]

    r0 = postprocess.mediancomp(scae[0])
    r1 = postprocess.mediancomp(scae[1])
    r2 = postprocess.mediancomp(scae[2])
    scaeticks = [r0,r1,r2]


    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,1,figsize=([10*cm,5*cm]),sharex=False,sharey=False,constrained_layout=True,dpi=1000)
    
    axs.plot(ticks,poderror,marker='.',lw=0.7,color='C1', ms=5)

    axs.plot(ticks,cnnaeerror,marker='.',lw=0.7,color='C2', ms=5)
    axs.plot(scaeticks,scaeerror,marker='.',lw=0.7,color='C3', ms=5)
    
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    minor_locator = AutoMinorLocator(2)
    axs.yaxis.set_minor_locator(minor_locator)
    axs.yaxis.set_major_locator(MultipleLocator(0.1))


    axs.grid(True)
    axs.grid(which='minor',alpha=0.2)
    axs.set_xscale('log', base=10)
    axs.legend(['POD','CNNAE','SCAE'],prop={'size': 6})
    
    #axs[0].set_title(name.capitalize(),weight="bold")
    axs.set_ylabel(r"$\ell_{{2}}$-error")
    axs.set_xlabel(r'Latent variables') 
    

    #Setting labels and stuff
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/errorplot_{}.pdf".format(domain,domain),bbox_inches='tight')
    #plt.show()

def uslice4(target,POD,CNNAE,SCAE,name,domain,dim,save=True):
    """"2D slice plot 4 scae mode
    Args:
        comp (array):  
        names (list): list of the names of the data. Normally train,validaiton,test
        name (str): name of save
        ds (xrray): 
        dim (str): dimension to slice
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib 
    import seaborn as sns
    import xarray as xr
    import numpy as np
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
    ds=ds.isel(y=slice(0, 32))
    #x, y = np.meshgrid(ds['x'].values, ds['y'].values)
    x = ds['x'].values
    y = ds['y'].values
    z = ds['z'].values

    u_tau = 0.05
    nu = 0.0004545454545
    
    x_plus = x*(u_tau/nu)
    y_plus = abs(y-y.max())*(u_tau/nu) #y_plus
    z_plus = (z + (-z[0]))*(u_tau/nu)
    xtitle = r'$x^{+}$'
    if dim == 'y':
    #Create meshgrid
        x, y = np.meshgrid(x_plus, y_plus, indexing='xy')
    elif dim == 'z':
        x, y = np.meshgrid(x_plus, z_plus, indexing='xy')
    

    time = 100 #original
    #time = 491 #KE_max_test
    #time = 11 #KE_min_test
    if dim == 'y':
        z1 = target[time,:,:,16,0].T
        z2 = POD[time,:,:,16,0].T
        z3 = CNNAE[time,:,:,16,0].T
        z4 = SCAE[time,:,:,16,0].T
        print("z+={}".format(z_plus[16]))

    if dim == 'z':
        z1 = target[time,:,16,:,0].T
        z2 = POD[time,:,16,:,0].T
        z3 = CNNAE[time,:,16,:,0].T
        z4 = SCAE[time,:,16,:,0].T
        print("y+={}".format(y_plus[16]))

    cm = 1/2.54  # centimeters in inches

    fig, axs = plt.subplots(2,2,figsize=([12*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)
    vmin = min([z1.min(), z2.min(), z3.min(), z4.min()])
    vmax = max([z1.max(), z2.max(), z3.max(), z4.max()])
    max_ax = np.argmax([z1.max(), z2.max(), z3.max(), z4.max()]) #if time scale colorbar pcm to the axs containing highest z value
    print(max_ax)
    #Mode 1
    pcm=axs[0,0].contourf(x,y,z1,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    axs[0,0].set_title('Target',weight="bold") #title
    axs[0,0].set_ylabel(r'${}^{{+}}$'.format(dim))

    #Mode 2
    axs[1,0].contourf(x,y,z2,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    axs[1,0].set_xlabel(r'$x^{+}$')
    axs[1,0].set_ylabel(r'${}^{{+}}$'.format(dim))
    axs[1,0].set_title('POD',weight="bold") #title
    #Mode 3
    axs[0,1].contourf(x,y,z3,levels=200,cmap='jet', vmin=vmin, vmax=vmax)

    axs[0,1].set_title('CNNAE',weight="bold") #title

    #Mode 4
    axs[1,1].contourf(x,y,z4,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    axs[1,1].set_xlabel(r'$x^{+}$')
    axs[1,1].set_title('SCAE',weight="bold") #title

    #Decide which one to use for colorbar
    if max_ax == 0:
        pcm=axs[0,0].contourf(x,y,z1,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    elif max_ax == 1:
        pcm=axs[1,0].contourf(x,y,z2,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    elif max_ax == 2:
        pcm=axs[0,1].contourf(x,y,z3,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    elif max_ax == 3:
        pcm=axs[1,1].contourf(x,y,z4,levels=200,cmap='jet', vmin=vmin, vmax=vmax)

    pcm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin,vmax),cmap='jet')
    #Remove annoying white lines
    for ax in axs.reshape(-1): 
        for c in ax.collections:
            c.set_edgecolor("face")
    
    fig.subplots_adjust(hspace=0.38) 
    ticks = np.linspace(vmin, vmax, 5, endpoint=True)
    cbar=fig.colorbar(pcm,ax=axs[:,:],aspect=30,shrink=0.7,location="bottom",pad=0.18,
    format="%.2f",spacing='uniform',ticks=ticks)
    #cbar.formatter.set_powerlimits((0, 0))
    ticks = np.linspace(vmin, vmax, 5, endpoint=True)
    #cbar.set_ticks([-5 -2.5, 0.0, 2.5, 5]), cbar.set_ticklabels([-5 -2.5, 0.0, 2.5, 5])
    print(ticks)
    #cbar.set_ticks(ticks), cbar.set_ticklabels(ticks)
    cbar.ax.set_xlabel(r"$u^{'+}$",rotation=0)
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/uslice4_{}_{}.pdf".format(domain,name,dim),bbox_inches='tight')
    plt.show()

def domainerror(save=True):
    """Error plot
    Args:
    domain (str): The domain on which the POD has been carried out.
    scae (list): name of scae domains to determine latent space size
    Returns:

    """
    import os
    from DataHandling.features import slices
    import shutil
    import numpy as np
    from tensorflow import keras
    from DataHandling import POD
    from DataHandling import postprocess
    from DataHandling.features import preprocess
    import xarray as xr
    import matplotlib.pyplot as plt
    
    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,1,figsize=([10*cm,5*cm]),sharex=False,sharey=False,constrained_layout=True,dpi=1000)
    
    #Define domain
    domains = ['blonigan','nakamura','1pi']
    for i in domains:
        domain = i
        print('Processing'+' '+domain)


        ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
        ds=ds.isel(y=slice(0, 32)) #Reduce y-dim from 65 to 32 as done by nakamura
        #%%
        # Load model prediction/target - # from least to most compressed
        if domain == 'nakamura':
            name=["cosmic-feather-29","valiant-river-31","deep-leaf-32"]
        elif domain == 'blonigan':
            name=["swift-sky-34","volcanic-gorge-35","generous-flower-36"]
            scae=["dulcet-armadillo-70","fragrant-flower-71",'dandy-bush-79','northern-capybara-82'] #here 4 
        elif domain == '1pi':
            name=["ethereal-snow-37","confused-waterfall-38","noble-wind-39"]

        predlist=[]
        predscae=[]
        for i,j in zip(name,scae):
            model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(i))
            pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(i))
            predlist.append(pred["test"])

            model=keras.models.load_model("/home/au569913/DataHandling/models/trained/{}".format(j))
            pred = np.load('/home/au569913/DataHandling/models/output/{}/predictions.npz'.format(j))
            predscae.append(pred["test"])

        targ = np.load('/home/au569913/DataHandling/models/output/{}/targets.npz'.format(name[0]))   

        target=targ["test"] 

        # %% import POD
        modes = [1536, 192, 24]
        if domain == '1pi':
            modes = [3072,384,48]

        c1,d1 = POD.projectPOD(modes[0],domain)
        c2,d2 = POD.projectPOD(modes[1],domain)
        c3,d3 = POD.projectPOD(modes[2],domain)
        c = [c1,c2,c3]
        d = [d1,d2,d3]
        test_snap = np.load("/home/au569913/DataHandling/models/POD/{}/test_snap.npy".format(domain))

        name = ''
        modes = 400

        if domain == '1pi':
            ticks = [3072,384,48]
        else:
            ticks = [1536,192,24]

        scaeticks = []
        for i in range(0,3):
            scaeticks.append(postprocess.mediancomp(scae[i]))
    
        CNNAE_error=[]
        SCAE_error=[]
        POD_error=[]
        for i in range(0,3):
            CNNAE_error.append(postprocess.errornorm(predlist[i],target))
            if domain=='blonigan':
                SCAE_error.append(postprocess.errornorm(predscae[i],target))
            POD_error.append(postprocess.errornorm(c[i],test_snap))

        markerdict = {'blonigan':'o','nakamura':'^','1pi':'s'}
        axs.plot(ticks,POD_error,marker=markerdict[domain],lw=0.7,color='C1', ms=5,fillstyle='none')
        axs.plot(ticks,CNNAE_error,marker=markerdict[domain],lw=0.7,color='C2', ms=5, fillstyle='none')
        #if domain == 'blonigan':
            #axs.plot(scaeticks,SCAE_error,marker=markerdict[domain],lw=0.7,color='C3', ms=5, fillstyle='none')

    axs.grid(True)
    axs.set_xscale('log', base=10)
    axs.legend(['POD','CNNAE','SCAE'], prop={'size': 6})
    #axs[0].set_title(name.capitalize(),weight="bold")
    axs.set_ylabel(r"$\ell_{{2}}$-error")
    axs.set_xlabel(r'Latent varibles') 


        #Setting labels and stuff
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/errorplot.pdf".format(domain),bbox_inches='tight')
    plt.show()

def CNNAEmode(CNNAE,target,name,domain,dim,save=True):
    """"2D slice plot 4 CNNAEmode
    Args:
        CNNAE (array): cnnae
        names (list): list of the names of the data. Normally train,validaiton,test
        name (str): name of save
        dim (str): dimension to slice
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib 
    import seaborn as sns
    import xarray as xr
    import numpy as np
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    ds=xr.open_zarr("/home/au569913/DataHandling/data/interim/{}.zarr".format(domain))
    ds=ds.isel(y=slice(0, 32))
    #x, y = np.meshgrid(ds['x'].values, ds['y'].values)
    x = ds['x'].values
    y = ds['y'].values
    z = ds['z'].values

    u_tau = 0.05
    nu = 0.0004545454545
    
    x_plus = x*(u_tau/nu)
    y_plus = abs(y-y.max())*(u_tau/nu) #y_plus
    z_plus = (z + (-z[0]))*(u_tau/nu)

    x_plus2 = np.append(x_plus[0:28:4], x_plus[31])
    y_plus2 = np.append(y_plus[0:28:4], y_plus[31])
    z_plus2 = np.append(z_plus[0:28:4], z_plus[31])

    xtitle = r'$x^{+}$'
    if dim == 'y':
    #Create meshgrid
        x, y = np.meshgrid(x_plus, y_plus, indexing='xy')
        x2, y2 = np.meshgrid(x_plus2, y_plus2, indexing='xy')
    elif dim == 'z':
        x, y = np.meshgrid(x_plus, z_plus, indexing='xy')
        x2, y2 = np.meshgrid(x_plus2, z_plus2, indexing='xy')
    
    mid = 16
    mid2 = 4

    time = 100 #original
    #time = 491 #KE_max_test
    #time = 11 #KE_min_test
    if dim == 'y':
        z1 = CNNAE[time,:,:,mid2,0].T
        z2 = target[time,:,:,mid,0].T
        print("z+={}".format(z_plus[mid]))

    if dim == 'z':
        z1 = CNNAE[time,:,mid2,:,0].T
        z2 = target[time,:,mid,:,0].T
        print("y+={}".format(y_plus[mid]))

    cm = 1/2.54  # centimeters in inches

    fig, axs =plt.subplots(1,2,figsize=([16*cm,7*cm]),sharex=False,sharey=True,constrained_layout=False,dpi=1000)
    vmin = min([z1.min()])
    vmax = max([z1.max()])
    #fig.subplots_adjust(hspace=0.38) 
    #Mode 1
    pcm=axs[0].contourf(x2,y2,z1,levels=200,cmap='jet', vmin=vmin, vmax=vmax)
    axs[0].set_title('Encoded CNNAE space',weight="bold") #title
    axs[0].set_ylabel(r'${}^{{+}}$'.format(dim))
    axs[0].set_xlabel(r'$x^{+}$')

    #Mode 2
    axs[1].contourf(x,y,z2,levels=200,cmap='jet', vmin=z2.min(), vmax=z2.max())
    axs[1].set_title('Decoded prediction',weight="bold") #title
    axs[1].set_xlabel(r'$x^{+}$')



    pcm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin,vmax),cmap='jet')
    #Remove annoying white lines
    for ax in axs:
        for c in ax.collections:
            c.set_edgecolor("face")
    
    #fig.subplots_adjust(hspace=0.38) 
    ticks = np.linspace(vmin, vmax, 5, endpoint=True)
    cbar=fig.colorbar(pcm,ax=axs[0],aspect=30,shrink=0.7,orientation="horizontal",pad=0.22,
    format="%.2f",spacing='uniform',ticks=ticks)
    
    print(ticks)


    pcm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(z2.min(), z2.max()),cmap='jet')
    ticks = np.linspace(z2.min(), z2.max(), 5, endpoint=True)
    cbar=fig.colorbar(pcm,ax=axs[1],aspect=30,shrink=0.7,orientation="horizontal",pad=0.22,
    format="%.2f",spacing='uniform',ticks=ticks)
    cbar.ax.set_xlabel(r"$u^{'+}$",rotation=0)
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/CNNAEmode_{}_{}.pdf".format(domain,name,dim),bbox_inches='tight')
    plt.show()

def SCAElatent(comp,domain,name,save=True):
    import matplotlib.pyplot as plt
    import numpy as np
    lan_var = []
    for i in range(len(comp)): #499
        lan_var.append(np.count_nonzero(comp[i,:,:,:,:])) #last 0/1/2 seems to be the same 18851ish
    
    fig = plt.figure()
    plt.plot(np.arange(0,499,1),lan_var,color='k',lw=0.3) # plot lantent variables
    plt.xlabel('Test snapshot')
    plt.ylabel('# of nonzero latent variables')
    plt.grid(True)
    
    from statistics import mean, median 
    mean_var = mean(lan_var)
    median_var = median(lan_var)

    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/SCAElatent_{}.pdf".format(domain,name),bbox_inches='tight')
    return median_var

def diss_arrangeplot(KE_total, KE_pred_total, KE_c3, KE_scae,domain,showscae=True,save=True):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from DataHandling import postprocess
    u_tau = 0.05

    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,figsize=([17*cm,7*cm]),sharex=True,sharey=False,constrained_layout=True,dpi=1000)

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

    #arr1inds = KE_total.isel(time=test_ind).values.argsort() # pick out indexes of sorted ds
    #plt.plot(np.arange(0,499,1),KE_total.isel(time=test_ind).values[arr1inds[::-1]],color='k')
    if type(KE_total) == xr.core.dataset.Dataset:
        arr1inds = KE_total.values.argsort() # pick out indexes of sorted ds
        plt.plot(np.arange(0,499,1),KE_total.values[arr1inds[::-1]],lw=2)
    else: 
        arr1inds = KE_total.argsort()
        plt.plot(np.arange(0,499,1),KE_total[arr1inds[::-1]],lw=2)

    plt.scatter(np.arange(0,499,1),KE_c3[arr1inds[::-1]]/(u_tau**2),marker='.',s=8,color='C1')
    plt.scatter(np.arange(0,499,1),KE_pred_total[arr1inds[::-1]],marker='.',s=8,color='C2')
    if showscae == True:
        plt.scatter(np.arange(0,499,1),KE_scae[arr1inds[::-1]],marker='.',s=8,color='C3')
    plt.ylabel(r'$Dissipation$')
    plt.legend(['DNS','POD','CNNAE','SCAE'])
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/diss_arranged_{}.pdf".format(domain, domain),bbox_inches='tight')

def diss_arrangeerror(KE_total, KE_pred_total, KE_c3, KE_scae,domain,showscae=True,save=True):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from DataHandling import postprocess
    u_tau = 0.05

    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,figsize=([17*cm,7*cm]),sharex=True,sharey=False,constrained_layout=True,dpi=1000)

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

    #arr1inds = KE_total.isel(time=test_ind).values.argsort() # pick out indexes of sorted ds
    #plt.plot(np.arange(0,499,1),KE_total.isel(time=test_ind).values[arr1inds[::-1]],color='k')
    if type(KE_total) == xr.core.dataset.Dataset:
        arr1inds = KE_total.values.argsort() # pick out indexes of sorted ds
        KE_total = KE_total.values
    
    else: 
        arr1inds = KE_total.argsort()
    plt.plot(np.arange(0,499,1),KE_total[arr1inds[::-1]]-KE_c3[arr1inds[::-1]]/(u_tau**2),lw=1,color='C1')
    plt.plot(np.arange(0,499,1),KE_total[arr1inds[::-1]]-KE_pred_total[arr1inds[::-1]],lw=1,color='C2')
    if showscae == True:
        plt.plot(np.arange(0,499,1),abs(KE_total[arr1inds[::-1]]-KE_scae[arr1inds[::-1]]),lw=1,color='C3')


    plt.ylabel(r'$Dissipation error$')
    plt.legend(['POD','CNNAE','SCAE'])
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/diss_arrangederror_{}.pdf".format(domain, domain),bbox_inches='tight')

def diss_arrangeplot(KE_total, KE_pred_total, KE_c3, KE_scae,domain,showscae=True,save=True):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from DataHandling import postprocess
    u_tau = 0.05

    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,figsize=([17*cm,7*cm]),sharex=True,sharey=False,constrained_layout=True,dpi=1000)

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

    #arr1inds = KE_total.isel(time=test_ind).values.argsort() # pick out indexes of sorted ds
    #plt.plot(np.arange(0,499,1),KE_total.isel(time=test_ind).values[arr1inds[::-1]],color='k')
    if type(KE_total) == xr.core.dataset.Dataset:
        arr1inds = KE_total.values.argsort() # pick out indexes of sorted ds
        plt.plot(np.arange(0,499,1),KE_total.values[arr1inds[::-1]],lw=2)
    else: 
        arr1inds = KE_total.argsort()
        plt.plot(np.arange(0,499,1),KE_total[arr1inds[::-1]],lw=2)

    plt.scatter(np.arange(0,499,1),KE_c3[arr1inds[::-1]]/(u_tau**2),marker='.',s=8,color='C1')
    plt.scatter(np.arange(0,499,1),KE_pred_total[arr1inds[::-1]],marker='.',s=8,color='C2')
    if showscae == True:
        plt.scatter(np.arange(0,499,1),KE_scae[arr1inds[::-1]],marker='.',s=8,color='C3')
    plt.ylabel(r'$Dissipation$')
    plt.legend(['DNS','POD','CNNAE','SCAE'])
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/diss_arranged_{}.pdf".format(domain, domain),bbox_inches='tight')

def TKE_arrangeerror(KE_total, KE_pred_total, KE_c3, KE_scae,domain,showscae=True,save=True):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from DataHandling import postprocess
    u_tau = 0.05

    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,figsize=([17*cm,7*cm]),sharex=True,sharey=False,constrained_layout=True,dpi=1000)

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

 
    POD_err = abs(KE_total-KE_c3/(u_tau**2))
    POD_arrange = POD_err.argsort()
    CNNAE_err = abs(KE_total-KE_pred_total)

    plt.plot(np.arange(0,499,1),POD_err[POD_arrange[::-1]],lw=1,color='C1')
    plt.plot(np.arange(0,499,1),CNNAE_err[POD_arrange[::-1]],lw=1,color='C2')
    if showscae == True:
        SCAE_err=abs(KE_total-KE_scae)
        plt.plot(np.arange(0,499,1),SCAE_err[POD_arrange[::-1]],lw=1,color='C3')

    axs.grid(True)
    plt.ylabel(r'$TKE^{+}$ error')
    plt.legend(['POD','CNNAE','SCAE'])
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/TKE_arrangederror_{}.pdf".format(domain, domain),bbox_inches='tight')

def errorcorr(KE_total, KE_pred_total, KE_c3, KE_scae, domain, showscae=True,save=True):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from DataHandling import postprocess
    u_tau = 0.05

    cm = 1/2.54  # centimeters in inches
    fig, axs=plt.subplots(1,2,figsize=([17*cm,7*cm]),sharex=False,sharey=True,constrained_layout=True,dpi=1000)

    test_ind =np.load("/home/au569913/DataHandling/data/interim/test_ind.npy")

 
    POD_err = abs(KE_total-KE_c3/(u_tau**2))
    POD_arrange = POD_err.argsort()
    CNNAE_err = abs(KE_total-KE_pred_total)
    SCAE_err=abs(KE_total-KE_scae)

    axs[0].scatter(CNNAE_err[POD_arrange[::-1]],POD_err[POD_arrange[::-1]],lw=1,color='C2',s=1)
    axs[1].scatter(SCAE_err[POD_arrange[::-1]],POD_err[POD_arrange[::-1]],lw=1,color='C3',s=1)

    axs[0].plot(np.array([0,CNNAE_err.max()]),np.array([0,CNNAE_err.max()]),color='k',lw=0.5)
    axs[1].plot(np.array([0,SCAE_err.max()]),np.array([0,SCAE_err.max()]),color='k',lw=0.5)

    axs[0].set_xlim([0, CNNAE_err.max()])
    axs[0].set_ylim([0, POD_err.max()])
    axs[1].set_xlim([0, SCAE_err.max()])
    axs[1].set_ylim([0, POD_err.max()])

    axs[0].grid(True)
    axs[1].grid(True)

    axs[0].set_xlabel(r'CNNAE error')
    axs[0].set_ylabel(r'POD error')
    axs[1].set_xlabel(r'SCAE error')
    if save == True:
        plt.savefig("/home/au569913/DataHandling/reports/{}/errorcorr_{}.pdf".format(domain, domain),bbox_inches='tight')