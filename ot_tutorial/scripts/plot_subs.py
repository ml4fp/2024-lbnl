import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import utils
import h5py as h5
import uproot3 as uproot
utils.SetStyle()


def GetBootErr(vals,quantile_list,n=100):
    res = []
    for i in range(n):
        boot = np.random.choice(vals,vals.shape[0])
        qt = np.quantile(boot,quantile_list)
        res.append((qt[1]-qt[0])*0.5)
    return np.std(res)


def PlotResolution(data_dict,plot_folder,gen_pt):

    ref_plot = 'gen_jet'
    feed_dict = {}
    
    pt_binning = np.geomspace(500,1500,8)
    thrust_res = {}
    thrust_err={}

    
    label_names={}    
    quantile_list = [0.25,0.75]

    binning = np.linspace(-1,1,20)
    feed_dict = {}
    groups = ['truth','abc','puppi']

    for key in groups:
        thrust_res[key]=np.zeros(pt_binning.shape[0]-1)
        thrust_err[key]=np.zeros(pt_binning.shape[0]-1)
        
        feed_dict[key] =  data_dict[key]
        quantiles = np.quantile(feed_dict[key],quantile_list)
        label_names[key] = "{}: {:.3f}".format(utils.name_translate[key],
                                                (quantiles[1]-quantiles[0])*0.5)
        
        
        for i in range(pt_binning.shape[0]-1):
            mask = (gen_pt>pt_binning[i]) & (gen_pt<pt_binning[i+1])
            quantiles = np.quantile(np.nan_to_num(data_dict[key][mask]),quantile_list)
            thrust_res[key][i] = (quantiles[1]-quantiles[0])*0.5
            thrust_err[key][i] = GetBootErr(np.nan_to_num(data_dict[key][mask]),quantile_list)

    pt_x = 0.5*(pt_binning[:-1] + pt_binning[1:])
    fig,ax0 = utils.PlotRoutine(thrust_res,xlabel=r'Jet p$_{T}$ [GeV]', ylabel= r'Jet $\tau_3/\tau_2$ resolution',plot_ratio=True,xaxis=pt_x,yerror=thrust_err,reference_name='puppi')    
    fig.savefig('{}/resolution_tau32.pdf'.format(plot_folder),bbox_inches='tight')

    fig,ax0 = utils.HistRoutine(feed_dict,xlabel=r'Jet $\tau_3/\tau_2$ resolution',
                                ylabel= 'Normalized entries',plot_ratio=False,binning=binning,
                                label_names=label_names)    
    fig.savefig('{}/resolution_tau32_inc.pdf'.format(plot_folder), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/PU/PU/vertex_info', help='Folder containing data and MC files')        
    #parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/PU/vertex_info', help='Folder containing data and MC files')
    parser.add_argument('--model', default=None, help='model checkpoint to load')
    parser.add_argument('--file', default="zprime_emd.root", help='dataset to load')
    flags = parser.parse_args()
    
    plot_folder = os.path.join("..","plots_{}".format(flags.model))

    file_path = os.path.join(flags.data_folder,flags.file)
    temp_file = uproot.open(file_path)['events']
    gen_pt = temp_file['genjetpt'].array()
    
    mask = (temp_file['genjetpt'].array()>0)*(temp_file['truthjetpt'].array()>0)*(temp_file['puppijetpt'].array()>0)*(temp_file['abcjetpt'].array()>0) #keep only jets matched to gen
    
    gen_pt = gen_pt[mask]
    
    groups = ['puppi','abc','truth']
    
    feed_dict = {}
    for group in groups:
        tau32 = temp_file['{}jettau3'.format(group)].array()/temp_file['{}jettau2'.format(group)].array()
        gen_tau32 = temp_file['genjettau3'].array()/temp_file['genjettau2'].array()
        feed_dict[group] = np.ma.divide(tau32-gen_tau32,gen_tau32).filled(0)
        feed_dict[group]=feed_dict[group][mask]
        
    PlotResolution(feed_dict,plot_folder,gen_pt)
        
