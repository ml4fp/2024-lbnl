import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import argparse
import h5py as h5
import utils
from ABCNet import ABCNet, SWD
import fastjet
import awkward as ak
import itertools
import matplotlib.pyplot as plt


def SaveH5(weights,pu_part,nopu_part):
    with h5.File(os.path.join(flags.data_folder,"Small_{}.h5".format(dataset)),"w") as h5f:
        dset = h5f.create_dataset('ABCNet', data=weights)
        dset = h5f.create_dataset('pu', data=pu_part)
        dset = h5f.create_dataset('no_pu', data=nopu_part)
        
    input("Saved")

def get_thrust(jets,parts):
    
    part_pt = np.sqrt(parts["px"]**2 + parts["py"]**2)
    part_phi = np.arctan2(parts["py"],parts["px"])
    part_eta = np.arcsinh(parts["pz"]/part_pt)
    z = part_pt/jets['pt']
    dr = (part_eta-jets['eta'])**2 + (part_phi-jets['phi'])**2
    thrust = np.sum(dr*z,-1)
    return thrust

def Plot2D(weights,pu,gen,checkpoint,name=''):
    utils.SetStyle()
    eta_binning = np.linspace(-4,4,25)
    phi_binning = np.linspace(-3.1,3.1,25)
    #pu = pu_part*weights
    plot_folder = os.path.join("..","plots_{}".format(checkpoint))
    etaphi_frac = np.zeros((eta_binning.shape[0]-1,phi_binning.shape[0]-1))
    etaphi_frac_n = np.zeros((eta_binning.shape[0]-1,phi_binning.shape[0]-1))
    etaphi_frac_c = np.zeros((eta_binning.shape[0]-1,phi_binning.shape[0]-1))

    eta_x = 0.5*(eta_binning[:-1] + eta_binning[1:])
    phi_x = 0.5*(phi_binning[:-1] + phi_binning[1:])


    etaphi_frac, xedges, yedges = np.histogram2d(pu[:,:,0].flatten(), pu[:,:,1].flatten(),
                                                 weights=(weights*pu[:,:,3]).flatten(), bins=(eta_binning, phi_binning))
    gen_frac,_,_ = np.histogram2d(gen[:,:,0].flatten(), gen[:,:,1].flatten(), weights=gen[:,:,3].flatten(),bins=(eta_binning, phi_binning))
    etaphi_frac/=gen_frac
    
    etaphi_frac_n, xedges, yedges = np.histogram2d(pu[:,:,0].flatten(), pu[:,:,1].flatten(),
                                                 weights=(weights*pu[:,:,3]*(pu[:,:,-2]==0)).flatten(), bins=(eta_binning, phi_binning))
    gen_frac,_,_ = np.histogram2d(gen[:,:,0].flatten(), gen[:,:,1].flatten(), weights=(gen[:,:,3]*(gen[:,:,-2]==0)).flatten(),bins=(eta_binning, phi_binning))
    etaphi_frac_n/=gen_frac
    
    etaphi_frac_c, xedges, yedges = np.histogram2d(pu[:,:,0].flatten(), pu[:,:,1].flatten(),
                                                 weights=(weights*pu[:,:,3]*(pu[:,:,-2]!=0)).flatten(), bins=(eta_binning, phi_binning))
    gen_frac,_,_ = np.histogram2d(gen[:,:,0].flatten(), gen[:,:,1].flatten(), weights=(gen[:,:,3]*(gen[:,:,-2]!=0)).flatten(),bins=(eta_binning, phi_binning))
    etaphi_frac_c/=gen_frac



    feed_dict = {
        '{}'.format(name):etaphi_frac,
        'Charged_{}'.format(name):etaphi_frac_c,
        'Neutral_{}'.format(name):etaphi_frac_n,
        }

    name_translate={
        '{}'.format(name):name.split("_")[0],
        'Charged_{}'.format(name):'TOTAL',
        'Neutral_{}'.format(name):'TOTAL',
        }

    
    for sample in feed_dict:
        cmap = plt.get_cmap('RdBu')
        fig,gs = utils.SetGrid(False,figsize=(8,10))
        ax = plt.subplot(gs[0])
        
        im = ax.pcolormesh(phi_x, eta_x, feed_dict[sample], cmap=cmap,vmin=0.04,vmax=2.0)
        fig.colorbar(im, ax=ax,label=r'$E_{reco} / E_{0 PU}$')
        bar = ax.set_title(name_translate[sample])
        #ax.set_xscale('log')
        ax.set_ylabel(r'$\eta$',fontsize=20)
        ax.set_xlabel(r'$\phi$',fontsize=20)
        fig.savefig('{}/efrac_2D_{}.pdf'.format(plot_folder,sample))
    print("done")

def GetMET(parts):
    '''
    Inputs:
    data of shape (N,npart,4)
    Outputs:
    MET pt, MET phi: (N,2)
    '''

    MET = -np.sum(parts,1)
    MET_pt = np.sqrt(MET[:,0]**2+MET[:,1]**2)
    MET_phi = np.arctan2(MET[:,1],MET[:,0])
    return np.stack((MET_pt,MET_phi),-1)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()
        
    #parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/PU/vertex_info', help='Folder containing data and MC files')
    #parser.add_argument('--data_folder', default='/global/cscratch1/sd/vmikuni/PU/vertex_info', help='Folder containing data and MC files')
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/PU/PU/vertex_info', help='Folder containing data and MC files')
    parser.add_argument('--dataset', default=None, help='dataset to load')
    parser.add_argument('--model', default=None, help='model checkpoint to load')
    parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--config', default='config.json', help='Config file with training parameters')
    
    flags = parser.parse_args()
    dataset_config = utils.LoadJson(flags.config)
    preprocessing = utils.LoadJson(dataset_config['PREPFILE'])

    if flags.dataset is None:
        dataset = dataset_config['FILES'][0]
    else:
        dataset = flags.dataset
    NPART=dataset_config['NPART']


    
    if flags.model is None:
        checkpoint = dataset_config['CHECKPOINT_NAME']
    else:
        checkpoint = flags.model


    checkpoint_folder = '../checkpoints_{}'.format(checkpoint)
    data = utils.EvalLoader(os.path.join(flags.data_folder,dataset),flags.nevts)
    inputs,outputs = ABCNet(npoint=NPART,nfeat=dataset_config['SHAPE'][2])
    model = Model(inputs=inputs,outputs=outputs)
    model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint'))
    abcnet_weights = model.predict(utils.ApplyPrep(preprocessing,data['pu_part'][:,:NPART]),batch_size=5)

    #abcnet_weights[abcnet_weights<1e-3]=0

    #abcnet_weights = model.predict(data['pu_part'][:,:NPART],batch_size=10)
    puppi_weights = data['pu_part'][:,:NPART,-4]
    chs_weights = data['pu_part'][:,:NPART,-1]
    chs_weights[data['pu_part'][:,:NPART,-2]==0]=1

    Plot2D(np.squeeze(abcnet_weights),data['pu_part'],data['nopu_part'],checkpoint,name='ABCNet_{}'.format(dataset))
    Plot2D(puppi_weights,data['pu_part'],data['nopu_part'],checkpoint,name='PUPPI_{}'.format(dataset))
    # SaveH5(np.squeeze(abcnet_weights)[:10],data['pu_part'][:10],data['nopu_part'][:10])
    def _convert_kinematics(data,is_gen=False):
        four_vec = data[:,:,:3]
            
        #eta,phi,pT,E
        #convert to cartesian coordinates (px,py,pz,E)
        cartesian = np.zeros(four_vec.shape,dtype=np.float32)
        cartesian[:,:,0] += np.abs(four_vec[:,:,2])*np.cos(four_vec[:,:,1])
        cartesian[:,:,1] += np.abs(four_vec[:,:,2])*np.sin(four_vec[:,:,1])
        cartesian[:,:,2] += np.abs(four_vec[:,:,2])*np.ma.sinh(four_vec[:,:,0]).filled(0)
        # cartesian[:,:,3] = four_vec[:,:,3]
        #print(cartesian)
        return cartesian
    
    nopu_set = _convert_kinematics(data['nopu_part'])
    gen_set =  _convert_kinematics(data['gen_part'],is_gen=True)
    abc_set = _convert_kinematics(data['pu_part'][:,:NPART])*(abcnet_weights)
    puppi_set = _convert_kinematics(data['pu_part'][:,:NPART])*np.expand_dims(puppi_weights,-1)


    del data['nopu_part'], data['pu_part']
    
    dict_dataset = {}
    dict_dataset['NPV'] = data['high_level'][:,-1]
    dict_dataset['MET_gen'] = GetMET(gen_set)
    dict_dataset['MET_nopu'] = GetMET(nopu_set)
    dict_dataset['MET_puppi'] = GetMET(puppi_set)
    dict_dataset['MET_abc'] = GetMET(abc_set)
    
    def _cluster(data):
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
        #dumb conversion for inputs
        events = []
        for datum in data:
            events.append([{"px": part[0], "py": part[1], "pz": part[2], "E": (part[0]**2+part[1]**2+part[2]**2)**0.5} for part in datum if np.abs(part[0])>0] )
            
        array = ak.Array(events)
        cluster = fastjet.ClusterSequence(array, jetdef)
        jets = cluster.inclusive_jets(min_pt=15)
        part = cluster.constituents(min_pt=15)

        jets["pt"] = np.sqrt(jets["px"]**2 + jets["py"]**2)
        jets["phi"] = np.arctan2(jets["py"],jets["px"])
        jets["eta"] = np.arcsinh(jets["pz"]/jets["pt"])
        jets['thrust'] = get_thrust(jets,part)

        jets=fastjet.sorted_by_pt(jets) 
        return jets[:,::-1]

    sets = {'nopu_jet':nopu_set,'abc_jet':abc_set,'puppi_jet':puppi_set,'gen_jet':gen_set}

    def _dict_data(jets,njets):        
        cluster = _cluster(jets)
        dataset = np.zeros((len(cluster.pt.to_list()),njets,5),dtype=np.float32)

        dataset[:,:,0]+=np.array(list(itertools.zip_longest(*cluster.pt.to_list(), fillvalue=0))).T[:,:njets]
        dataset[:,:,1]+=np.array(list(itertools.zip_longest(*cluster.eta.to_list(), fillvalue=0))).T[:,:njets]
        dataset[:,:,2]+=np.array(list(itertools.zip_longest(*cluster.phi.to_list(), fillvalue=0))).T[:,:njets]
        dataset[:,:,3]+=np.array(list(itertools.zip_longest(*cluster.E.to_list(), fillvalue=0))).T[:,:njets]
        dataset[:,:,4]+=np.array(list(itertools.zip_longest(*cluster.thrust.to_list(), fillvalue=0))).T[:,:njets]
        return dataset
    
    for dset in sets:
        print(dset)
        dict_dataset[dset] = _dict_data(sets[dset],njets=9)
        
    with h5.File(os.path.join(flags.data_folder,"JetInfo_{}_{}".format(checkpoint,dataset)),"w") as h5f:
        for key in dict_dataset:
            dset = h5f.create_dataset(key, data=dict_dataset[key])
            
