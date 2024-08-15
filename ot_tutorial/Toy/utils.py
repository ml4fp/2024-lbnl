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
    'reco':'dotted',
    'sk':"-",
    
}

colors = {
    'nopu':'black',
    'truth':'black',
    'abc': '#7570b3',
    'puppi': "#d95f02",
    'reco':'black',  
    'sk':'#1b9e77',
}


name_translate={
    'nopu':"0 PU",
    'truth':"0 PU",
    'abc': "TOTAL",
    'puppi': "PUPPI",
    'reco':"Gen",
    'sk':"SK",
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
    ax0.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
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




def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)
        
def convert_preprocessing(data):
    mask = data[:,:,2]==0
    #data[:,:,2] = 200*data[:,:,2]*(mask==False)
    data[:,:,2] = np.exp(data[:,:,2])*(mask==False)
    return data

def preprocessing(file_name,data_value,nparts=1000,is_training=True):
    rank = hvd.rank()
    size = hvd.size()
    data_label = str(data_value)
    features = []    
    with h5.File(file_name,"r") as h5f:
        #for key in ['200']: #read just one mass point
        for key in [data_label]: #read just one mu point
        #for key in h5f:
            data = h5f[key][rank::size]
            padded = np.zeros((data.shape[0],nparts,data.shape[-1]))
            padded[:,:data.shape[1]]+=data[:,:nparts]
            features.append(padded)
   
    features = np.concatenate(features)
    jet_info = features[:,:,8:9]
    label = features[:,:,4]
    label = np.expand_dims(label,-1)
    features[:,:,2] -=np.pi #have phi between -pi/2,pi/2
    features=np.concatenate([features[:,:,:4],features[:,:,5:7]],-1)
    #normalize pt
    features[:,:,0]= np.ma.log(features[:,:,0]).filled(0)
    #change from (pt,eta,phi) to (eta,phi,pt)
    features[:,:,[0,1,2]] = features[:,:,[1,2,0]]                
    nopu = features*(label==0)
    #print(features,nopu)
         
    return features,nopu,jet_info,label
        

def convert_coordinate(data,mask):
    data_mask = data*mask

    px = (data_mask[:,:,2]*np.cos(data_mask[:,:,1]))[:,:,None]
    py = (data_mask[:,:,2]*np.sin(data_mask[:,:,1]))[:,:,None]
    pz = (data_mask[:,:,2]*np.sinh(data_mask[:,:,0]))[:,:,None]

    return px,py,pz
