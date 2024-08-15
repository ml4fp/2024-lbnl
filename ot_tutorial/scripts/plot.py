import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import utils
import h5py as h5
from scipy.spatial import distance_matrix

import tensorflow.keras.backend as K
utils.SetStyle()


def TakeClosest(ref,parts,idx=0,npv=None):
    '''
    Calculates the distance between objects from the parts dataset wrt the ref dataset.
    Returns the pt resolution of matched to gen jets with the coordinates of the matched gen jet in the format (pt resolution,pt,eta)
    '''
    dist = [distance_matrix(ref[i,:,1:3], parts[i,:,1:3]) for i in range(ref.shape[0])]
    index_array = np.argmin(np.array(dist),-1) #taking closest
    delta_r_mask = (np.min(np.array(dist),-1) < 0.4) & (np.min(np.array(dist),-1) > 0) #taking closest
    
    parts_ordered = np.take_along_axis(parts, np.expand_dims(index_array, axis=-1), axis=1)
    #resolution = np.ma.divide(parts_ordered[:,:,0]-ref[:,:,0],ref[:,:,0]).filled(-100)
    resolution = np.ma.divide(parts_ordered[:,:,idx]-ref[:,:,idx],ref[:,:,idx]).filled(-100)
    delta_r_mask = (delta_r_mask) & (resolution>-100) & (resolution>-0.95)
    if npv is not None:

        npv_tile = np.tile(npv,(1,resolution.shape[1]))

        resolution=np.stack([resolution.flatten()[delta_r_mask.flatten()],
                             ref[:,:,0].flatten()[delta_r_mask.flatten()],
                             ref[:,:,1].flatten()[delta_r_mask.flatten()],
                             ref[:,:,4].flatten()[delta_r_mask.flatten()],
                             npv_tile.flatten()[delta_r_mask.flatten()]
        ],-1)

    else:        
        resolution=np.stack([resolution.flatten()[delta_r_mask.flatten()],
                             ref[:,:,0].flatten()[delta_r_mask.flatten()],
                             ref[:,:,1].flatten()[delta_r_mask.flatten()],
                             ref[:,:,4].flatten()[delta_r_mask.flatten()],
        ],-1)
    #resolution=resolution.flatten()[delta_r_mask.flatten()]
    #print(resolution.shape)
    return resolution


def FindNPUJets(ref,parts):
    dist = [distance_matrix(ref[i,:,1:3], parts[i,:,1:3]) for i in range(ref.shape[0])]
    index_array = np.argmin(np.array(dist),-1) #taking closest
    delta_r_mask = (np.min(np.array(dist),-1) > 0.4) & (np.min(np.array(dist),-1) > 0) #taking closest
    
    parts_ordered = np.take_along_axis(parts, np.expand_dims(index_array, axis=-1), axis=1)
    resolution = np.ma.divide(parts_ordered[:,:,0]-ref[:,:,0],ref[:,:,0]).filled(-100)
    delta_r_mask = (delta_r_mask) & (resolution>-100)
    
    resolution=np.stack([resolution.flatten()[delta_r_mask.flatten()],
                         ref[:,:,0].flatten()[delta_r_mask.flatten()],
                         ref[:,:,1].flatten()[delta_r_mask.flatten()]],-1)
    #resolution=resolution.flatten()[delta_r_mask.flatten()]
    #print(resolution.shape)
    return resolution

def GetBootErr(vals,quantile_list,n=100):
    res = []
    for i in range(n):
        boot = np.random.choice(vals,vals.shape[0])
        qt = np.quantile(boot,quantile_list)
        res.append((qt[1]-qt[0])*0.5)
    return np.std(res)
        

def PlotResolution(data_dict,plot_folder,process,npv=None):
    def _preprocess(ref,data):
        preprocessed = np.array(TakeClosest(np.array(ref),np.array(data),npv=npv,idx=0))
        return preprocessed

    ref_plot = 'gen_jet'
    feed_dict = {}
    
    pt_binning = np.geomspace(15,300,8)
    pt_res = {}
    pt_err = {}
    eta_binning = np.linspace(0,4,5)
    eta_res = {}
    eta_err={}
    # thrust_binning = np.linspace(0,1,5)
    # thrust_res = {}
    # thrust_err={}

    
    npv_binning = np.linspace(0,200,1)
    npv_res = {}
    npv_err={}
    
    etapt_res = {}
    label_names={}
    
    quantile_list = [0.25,0.75]
    
    #Overall resolution
    for key in data_dict:
        if ref_plot in key:continue
        name = key.replace("_jet","")
        res = _preprocess(data_dict[ref_plot],data_dict[key])
        feed_dict[name] = res[:,0]
        quantiles = np.quantile(feed_dict[name],quantile_list)
        label_names[name] = "{}: {:.3f}".format(utils.name_translate[name],
                                                (quantiles[1]-quantiles[0])*0.5)
        
        
        pt_res[name]=np.zeros(pt_binning.shape[0]-1)
        pt_err[name]=np.zeros(pt_binning.shape[0]-1)
        eta_res[name]=np.zeros(eta_binning.shape[0]-1)
        eta_err[name]=np.zeros(eta_binning.shape[0]-1)
        # thrust_res[name]=np.zeros(eta_binning.shape[0]-1)
        # thrust_err[name]=np.zeros(eta_binning.shape[0]-1)
        npv_res[name]=np.zeros(npv_binning.shape[0]-1)
        npv_err[name]=np.zeros(npv_binning.shape[0]-1)
        
        etapt_res[name]= np.zeros((eta_binning.shape[0]-1,pt_binning.shape[0]-1))

        for i in range(npv_binning.shape[0]-1):
            mask = (res[:,4]>npv_binning[i]) & (res[:,4]<npv_binning[i+1])
            # print(i,np.sum(mask))
            # print(np.quantile(res[:,0][mask],q=[0.25,0.75]))
            quantiles = np.quantile(np.nan_to_num(res[:,0][mask]),quantile_list)
            npv_res[name][i] = (quantiles[1]-quantiles[0])*0.5
            npv_err[name][i] = GetBootErr(np.nan_to_num(res[:,0][mask]),quantile_list)

        
        for i in range(pt_binning.shape[0]-1):
            mask = (res[:,1]>pt_binning[i]) & (res[:,1]<pt_binning[i+1])
            # print(i,np.sum(mask))
            # print(np.quantile(res[:,0][mask],q=[0.25,0.75]))
            quantiles = np.quantile(np.nan_to_num(res[:,0][mask]),quantile_list)
            pt_res[name][i] = (quantiles[1]-quantiles[0])*0.5
            pt_err[name][i] = GetBootErr(np.nan_to_num(res[:,0][mask]),quantile_list)
            
            #pt_err[name][i] = 1.0/np.sqrt(np.sum(mask))
        for i in range(eta_binning.shape[0]-1):
            mask = (np.abs(res[:,2])>eta_binning[i]) & (np.abs(res[:,2])<eta_binning[i+1])

            quantiles = np.quantile(res[:,0][mask],quantile_list)
            eta_res[name][i] = (quantiles[1]-quantiles[0])*0.5
            eta_err[name][i] = GetBootErr(np.nan_to_num(res[:,0][mask]),quantile_list)

        # for i in range(thrust_binning.shape[0]-1):
        #     mask = (np.abs(res[:,3])>thrust_binning[i]) & (np.abs(res[:,3])<thrust_binning[i+1])

        #     quantiles = np.quantile(res[:,0][mask],quantile_list)
        #     thrust_res[name][i] = (quantiles[1]-quantiles[0])*0.5
        #     thrust_err[name][i] = GetBootErr(np.nan_to_num(res[:,0][mask]),quantile_list)


        for i in range(pt_binning.shape[0]-1):
            for j in range(eta_binning.shape[0]-1):
                mask = (np.abs(res[:,2])>eta_binning[j]) & (np.abs(res[:,2])<eta_binning[j+1])
                mask = mask & (res[:,1]>pt_binning[i]) & (res[:,1]<pt_binning[i+1])
                quantiles = np.quantile(res[:,0][mask],quantile_list)
                etapt_res[name][j,i] = (quantiles[1]-quantiles[0])*0.5

    binning=np.linspace(-1,1,20)
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Jet energy resolution', ylabel= 'Normalized entries',plot_ratio=False,binning=binning,label_names=label_names)    
    fig.savefig('{}/resolution_{}.pdf'.format(plot_folder,process), bbox_inches='tight')

    npv_x = 0.5*(npv_binning[:-1] + npv_binning[1:])
    fig,ax0 = utils.PlotRoutine(npv_res,xlabel=r'Number of primary vertices', ylabel= 'Jet energy resolution',plot_ratio=False,xaxis=npv_x,yerror=npv_err,figsize=(16,7))
    fig.savefig('{}/resolution_npv_{}.pdf'.format(plot_folder,process), bbox_inches='tight')

    pt_x = 0.5*(pt_binning[:-1] + pt_binning[1:])
    fig,ax0 = utils.PlotRoutine(pt_res,xlabel=r'Jet p$_{T}$ [GeV]', ylabel= 'Jet energy resolution',plot_ratio=True,xaxis=pt_x,yerror=pt_err,reference_name='puppi')
    fig.savefig('{}/resolution_pt_{}.pdf'.format(plot_folder,process), bbox_inches='tight')
    
    eta_x = 0.5*(eta_binning[:-1] + eta_binning[1:])
    fig,ax0 = utils.PlotRoutine(eta_res,xlabel=r'Jet $|\eta|$', ylabel= 'Jet energy resolution',plot_ratio=True,xaxis=eta_x,yerror=eta_err,reference_name='puppi')    
    fig.savefig('{}/resolution_eta_{}.pdf'.format(plot_folder,process), bbox_inches='tight')

    # thrust_x = 0.5*(thrust_binning[:-1] + thrust_binning[1:])
    # fig,ax0 = utils.PlotRoutine(thrust_res,xlabel=r'Jet thrust', ylabel= 'Jet energy resolution',plot_ratio=True,xaxis=thrust_x,yerror=thrust_err,reference_name='puppi')    
    # fig.savefig('{}/resolution_thrust_{}.pdf'.format(plot_folder,process))

    for key in etapt_res:
        cmap = plt.get_cmap('viridis')
        fig,gs = utils.SetGrid(False)
        ax = plt.subplot(gs[0])
        im = ax.pcolormesh(pt_x, eta_x, etapt_res[key], cmap=cmap,vmin=0.04,vmax=0.28)
        fig.colorbar(im, ax=ax,label='Jet energy resolution')
        bar = ax.set_title(key)
        ax.set_xscale('log')
        ax.set_xlabel(r'p$_{T}$ [GeV]',fontsize=20)
        ax.set_ylabel(r'$|\eta|$',fontsize=20)
        fig.savefig('{}/resolution_2D_{}_{}.pdf'.format(plot_folder,key,process), bbox_inches='tight')


    
def PlotNjet(data_dict,plot_folder,process):
    def _preprocess(data):
        mask = np.abs(data[:,:,1])<5
        preprocessed = np.sum((mask)*(data[:,:,0]>20),-1) #pt
        #print(preprocessed)
        return preprocessed


    feed_dict = {}
    for key in data_dict:
        feed_dict[key.replace("_jet","")] = _preprocess(data_dict[key])
    binning = list(range(1,8))
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Jet multiplicity', ylabel= 'Normalized entries',logy=True,binning=binning)
    fig.savefig('{}/njets_{}.pdf'.format(plot_folder,process), bbox_inches='tight')


def PlotNPUjet(data_dict,plot_folder):    
    def _preprocess(data):
        mask = np.abs(data[:,:,1])<5
        preprocessed = np.sum((mask)*(data[:,:,0]>20),-1) #pt
        #print(preprocessed)
        return preprocessed


    feed_dict = {}
    for key in data_dict:
        feed_dict[key.replace("_jet","")] = _preprocess(data_dict[key])


    fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Jet multiplicity', ylabel= 'Normalized entries',logy=True)
    fig.savefig('{}/njets_{}.pdf'.format(plot_folder,process), bbox_inches='tight')

    

def PlotMET(data_dict,plot_folder,process):    
    ref_plot = 'MET_gen'    

    feed_dict = {}
    for key in data_dict:
        feed_dict[key.replace("MET_","")] = data_dict[key][:,0]
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel=r'$p_{T}^{miss}$ [GeV]', ylabel= 'Normalized entries',logy=True,reference_name='gen',binning=np.linspace(10,300,10))
    fig.savefig('{}/met_pt_{}.pdf'.format(plot_folder,process), bbox_inches='tight')

    feed_dict = {}
    for key in data_dict:
        feed_dict[key.replace("MET_","")] = data_dict[key][:,1]
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel=r'MET $\phi$', ylabel= 'Normalized entries',logy=True,reference_name='gen',binning=np.linspace(-3,3,10))
    ax0.set_ylim([1e-2,1])
    fig.savefig('{}/met_phi_{}.pdf'.format(plot_folder,process), bbox_inches='tight')
    quantile_list = [0.25,0.75]

    feed_dict={}
    label_names={}
    
    for key in data_dict:
        if key == ref_plot:continue
        feed_dict[key.replace("MET_","")] = (data_dict[key][:,0]-data_dict[ref_plot][:,0])/data_dict[ref_plot][:,0]
        quantiles = np.quantile(feed_dict[key.replace("MET_","")],quantile_list)
        label_names[key.replace("MET_","")] = "{}: {:.3f}".format(utils.name_translate[key.replace("MET_","")],(quantiles[1]-quantiles[0])*0.5)
        
    binning=np.linspace(-2,2,20)
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel=r'$p_{T}^{miss}$ resolution', ylabel= 'Normalized entries',plot_ratio=False,binning=binning,label_names=label_names)
    
    
    fig.savefig('{}/met_pt_resolution_{}.pdf'.format(plot_folder,process), bbox_inches='tight')

    feed_dict={}
    label_names={}
    
    for key in data_dict:
        if key == ref_plot:continue
        feed_dict[key.replace("MET_","")] = (data_dict[key][:,1]-data_dict[ref_plot][:,1])/data_dict[ref_plot][:,1]
        quantiles = np.quantile(feed_dict[key.replace("MET_","")],quantile_list)
        label_names[key.replace("MET_","")] = "{}: {:.3f}".format(utils.name_translate[key.replace("MET_","")],(quantiles[1]-quantiles[0])*0.5)
        
    binning=np.linspace(-1,1,20)
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel='MET phi resolution', ylabel= 'Normalized entries',plot_ratio=False,binning=binning,label_names=label_names)    
    fig.savefig('{}/met_phi_resolution_{}.pdf'.format(plot_folder,process), bbox_inches='tight')


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/PU/PU/vertex_info', help='Folder containing data and MC files')        
    #parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/PU/vertex_info', help='Folder containing data and MC files')
    parser.add_argument('--dataset', default=None, help='dataset to load')
    parser.add_argument('--model', default=None, help='model checkpoint to load')

    parser.add_argument('--config', default='config.json', help='Basic config file containing general options')
    
    
    flags = parser.parse_args()
    dataset_config = utils.LoadJson(flags.config)

    if flags.dataset is None:
        sample = dataset_config['FILES'][0]
    else:
        sample = flags.dataset
        
    if flags.model is None:
        checkpoint = dataset_config['CHECKPOINT_NAME']
    else:
        checkpoint = flags.model

    file_name = os.path.join(flags.data_folder,"JetInfo_{}_{}".format(checkpoint,sample))

    sets = ['nopu_jet','abc_jet','puppi_jet','gen_jet']
    feed_dict = utils.loadSample(file_name,sets)    
    MET_set = ['MET_nopu','MET_abc','MET_puppi','MET_gen']
    feed_dict_met = utils.loadSample(file_name,MET_set)
    npv=utils.loadSample(file_name,['NPV'])
    plot_folder = os.path.join("..","plots_{}".format(checkpoint))
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
        

    plot_routines = {
        #'njets':PlotNjet,
        'pt resolution':PlotResolution,
    }
        
    for plot in plot_routines:
        print(plot)
        plot_routines[plot](feed_dict,plot_folder,sample.replace(".h5",""),npv=npv['NPV'])

    # met_routines = {
    #     'MET hist':PlotMET,
    #     }

    # for plot in met_routines:
    #     print(plot)
    #     met_routines[plot](feed_dict_met,plot_folder,sample.replace(".h5",""))
