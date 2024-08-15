import json, yaml
import os,copy
import h5py as h5
import horovod.tensorflow.keras as hvd
import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick


def split_data(data,nevts,frac=0.8):
    data = data.cache().shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    return train_data,test_data

line_style = {
    'nopu':'dotted',
    'truth':'dotted',
    'abc': "-",
    'puppi': "-",
    'gen':'dotted',
    
}

colors = {
    'nopu':'black',
    'truth':'black',
    'abc': '#7570b3',
    'puppi': "#d95f02",
    'gen':'#1b9e77',    
}


name_translate={
    'nopu':"0 PU",
    'truth':"0 PU",
    'abc': "TOTAL",
    'puppi': "PUPPI",
    'gen':"Gen",

}


def loadSample(file_name,sets):
    feed_dict = {}
    with h5.File(os.path.join(file_name),"r") as h5f:
        for dataset in sets:
            feed_dict[dataset] = h5f[dataset][:]
    return feed_dict


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(ratio=True,figsize=None):
    if figsize is None:
        figsize=(9,9)
    fig = plt.figure(figsize=figsize)
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs



def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='nopu',plot_ratio=False,xaxis=None,yerror=None,figsize=None):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid(ratio=plot_ratio,figsize=figsize) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):        
        if yerror is None:
            yerr = None
        else:
            yerr = yerror[plot]
            
        if xaxis is None:
            ax0.errorbar(feed_dict[plot],yerr=yerr,label=name_translate[plot],marker=line_style[plot],color=colors[plot])
        else:
            ax0.errorbar(xaxis,feed_dict[plot],yerr=yerr,label=name_translate[plot],marker='o',color=colors[plot])

        if reference_name!=plot and plot_ratio:
            if 'nopu' in plot or 'truth' in plot:continue
            ratio = 100*np.divide(-feed_dict[reference_name]+feed_dict[plot],feed_dict[reference_name])
            ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)

    #plt.axvline(x=140,color='r', linestyle='-',linewidth=1)
    ax0.legend(loc='best',fontsize=16,ncol=1)
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)

        plt.ylabel('Difference (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
        plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-50,50])
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='gen',logy=False,binning=None,label_loc='best',plot_ratio=True,weights=None,uncertainty=None,label_names=None):
    #assert reference_name in feed_dict.keys() and plot_ratio==True, "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    if label_names is None:
        label_names=name_translate
        
    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),10)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    if plot_ratio:
        reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    
    for ip,plot in enumerate(feed_dict.keys()):
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=label_names[plot],linestyle=line_style[plot],color=colors[plot],density=True,histtype="step",weights=weights[plot])
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=label_names[plot],linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
        
        if plot_ratio:
            if reference_name!=plot:
                ratio = 100*np.divide(reference_hist-dist,reference_hist)
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=1)        
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Difference. (%)')
        plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
        plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-50,50])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
        
    return fig,ax0


def DataLoader(file_name,nevts):
    rank = hvd.rank()
    size = hvd.size()
    with h5.File(file_name,"r") as h5f:
        pu = h5f['pu_part'][rank:int(nevts):size].astype(np.float32)
        nopu = h5f['nopu_part'][rank:int(nevts):size].astype(np.float32)          
    return pu,nopu

def EvalLoader(file_name,nevts):
    data_dict = {}
    
    with h5.File(file_name,"r") as h5f:
        for key in h5f:
            data_dict[key] = h5f[key][:nevts].astype(np.float32)
    return data_dict

def ApplyPrep(param_dict,data,use_log=False):
    shape = data.shape
    data_flat = data.copy().reshape((-1,shape[-1]))
    if use_log:
        data_flat[:,2]=np.ma.log(data_flat[:,2]).filled(0)
        data_flat[:,3]=np.ma.log(data_flat[:,3]).filled(0)
        data_flat[:,4]=np.ma.log(data_flat[:,4]).filled(0)
        data_flat[:,5]=np.ma.log(data_flat[:,5]).filled(0)
        
    #keep zeros
    mask = data_flat!=0
    data_flat = (data_flat-param_dict['mean'][:shape[-1]])/param_dict['std'][:shape[-1]]
    data_flat*=mask
    
    return data_flat.reshape(shape)

def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)

def Preprocess(name,raw_data):    
    '''Preprocess the data'''

    if 'PID' in name:
        unique_pid = [0,11,13,22,211,321,2212]
        #unique_pid = [-2212,-321,-211,-13,-11,11,13,22,211,321,2212]
        for i, unique in enumerate(unique_pid):
            raw_data[np.abs(raw_data)==unique] = i+1
        return raw_data/len(unique_pid)
    else:    
        return np.array(raw_data)

# def Preprocess(name,raw_data):
#     '''Preprocess the data'''
#     if 'Eta' in name or 'Phi' in name or 'PuppiW' in name or 'Charge' in name:
#         #print("nothing to do")
#         #no modification
#         return np.array(raw_data)
#     elif 'PT' in name or 'E' in name:
#         print("take log",name)
#         return np.ma.log10(raw_data).filled(0)
#     elif 'PID' in name:
#         unique_pid = [-2212,-321,-211,-13,-11,11,13,22,211,321,2212]
#         for i, unique in enumerate(unique_pid):
#             raw_data[raw_data==unique] = i+1
#         return raw_data/len(unique_pid)
#     else:
#         #D0, Dz
#         #print("log and sign")
#         return np.sign(raw_data)*np.ma.log10(np.abs(raw_data)).filled(0)/10.0
    
if __name__ == "__main__":
    #Preprocessing of the input files: conversion to cartesian coordinates + zero-padded mask generation
    import uproot3 as uproot
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--sample', default='ZJets', help='Physics sample to load')
    parser.add_argument('--base_path', default='/global/cfs/cdirs/m3929/PU/', help='Path to load files')
    parser.add_argument('--out_path', default='/pscratch/sd/v/vmikuni/PU/vertex_info', help='Path to load files')
    flags = parser.parse_args()
    
    #sample_name = 'DiJet'
    # sample_name = 'TTBar'
    #sample_name = 'WJets_HighPT'
    # sample_name = 'ZJets'
    # sample_name = 'VBFHinv'

    # base_path = '/global/cfs/cdirs/m3929/PU/'
    # base_path = '/pscratch/sd/v/vmikuni/PU/vertex_info'
    # out_path = '/pscratch/sd/v/vmikuni/PU/'
    # out_path = '/global/cscratch1/sd/vmikuni/PU'
    # out_path = '/pscratch/sd/v/vmikuni/PU/vertex_info'

    
    features = ['Eta','Phi','PT','E','D0','DZ','PuppiW','PID','Charge','hardfrac']
    pu_features = ["pu_pfs_{}".format(feat) for feat in features]
    nopu_features = ["nopu_pfs_{}".format(feat) for feat in features]
    genpart_branches = ['Eta','Phi','PT','E','Charge','PID']
    gen_info = ["nopu_gen_{}".format(gen) for gen in genpart_branches]
    high_level = ['nopu_genmet_MET','nopu_genmet_Phi','pu_npv_GenVertex_size']
    
    file_list = ['{}_outfile_{}.root'.format(flags.sample,i) for i in range(25,26)]
    
    merged_file = {}
    
    print("merging files")
    for sample in file_list:
        file_path = os.path.join(flags.base_path,sample)
        temp_file = uproot.open(file_path)['events']
        for feat in gen_info + nopu_features + pu_features + high_level:
            if feat in merged_file:
                merged_file[feat] = np.concatenate([merged_file[feat],temp_file[feat].array()],0)
            else:
                merged_file[feat] = temp_file[feat].array()
    del temp_file
    print("Preprocessing")
    def _merger(features,ndim=2):
        array=[]
        for feat in features:
            array.append(merged_file[feat])
        if ndim ==2:            
            return np.transpose(np.array(array).astype(np.float32),[1,0])
        else:
            return np.transpose(np.array(array).astype(np.float32),[1,2,0])
            


    
    high_array = _merger(high_level)
    gen_array = _merger(gen_info,ndim=3)
    neutrino_id = [12,14,16]
    mask = (np.abs(gen_array[:,:,-1]) == neutrino_id[0]) | (np.abs(gen_array[:,:,-1]) == neutrino_id[1]) | (np.abs(gen_array[:,:,-1]) == neutrino_id[2])

    #Mask neutrinos out from gen particles
    gen_array[mask]=0

    
    #Preprocess training data prior to training
    for feat in nopu_features + pu_features:
        merged_file[feat] = Preprocess(feat,merged_file[feat])

    #Fix the PID for 0-padded particles
    nopu_array = _merger(nopu_features,ndim=3)
    nopu_array[(nopu_array[:,:,-3]==1.0/7)&(nopu_array[:,:,0]==0.0)]=0
    pu_array = _merger(pu_features,ndim=3)
    pu_array[(pu_array[:,:,-3]==1.0/7)&(pu_array[:,:,0]==0.0)]=0

    
    #Apply CHS Only to charged particles
    # pu_array[:,:,-1]*=np.abs(pu_array[:,:,-2])
    # nopu_array[:,:,-1]*=np.abs(nopu_array[:,:,-2])
    
    with h5.File(os.path.join(flags.out_path,'{}_raw.h5'.format(flags.sample)),'w') as fh5:
        dset = fh5.create_dataset('high_level', data=high_array)
        dset = fh5.create_dataset('gen_part', data=gen_array)
        dset = fh5.create_dataset('pu_part', data=pu_array)
        dset = fh5.create_dataset('nopu_part', data=nopu_array)
                
