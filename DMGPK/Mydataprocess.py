import warnings
import torch
import torch.nn.functional as F
import numpy as np
import h5py
from nilearn import connectome
import scipy.io as sio
import sys
import argparse
import pandas as pd
import deepdish as dd
import os
warnings.filterwarnings("ignore")

#  compute connectivity matrices
def subject_connectivity(timeseries, kind):
    """
        timeseries   : timeseries table for subject (timepoints x regions) (85*524)
        subjects     : subject IDs
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        iter_no      : tangent connectivity iteration number for cross validation evaluation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder
    returns:
        connectivity : connectivity matrix (regions x regions) 
    """

    if kind in ['TPE', 'TE', 'correlation','partial correlation']:
        if kind not in ['TPE', 'TE']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform(timeseries)
        else:
            if kind == 'TPE':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                conn_mat = conn_measure.fit_transform(timeseries)
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(conn_mat)
                connectivity = connectivity_fit.transform(conn_mat)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(timeseries)
                connectivity = connectivity_fit.transform(timeseries)

    return connectivity
    

if __name__ == '__main__':
    # main() 
    # change the save paths
    # Compute and save connectivity matrices
    
    feature_matrix = np.transpose(h5py.File('D:/ResearchGroup/ProjectCode/DMGPK/MyData/NC_AD/NC_AD.mat', 'r')['NC_AD'][()])
    labely = np.transpose(h5py.File('D:/ResearchGroup/ProjectCode/DMGPK/MyData/NC_AD/label_ND.mat', 'r')['label_ND'][()])
    
    ### ------------- ROI/GENE feature matrix 246*246 ----------------------###
    # ROI_matrix = feature_matrix[:,0:246,:]
    # Gene_matrix = feature_matrix[:,246:525,:]
    # max_len = 246
    # dim = 85
    # ### ------------- Gene/ROI feature matrix ----------------------###
    # feature_matrix2 = ROI_matrix.reshape((172,85,246))
    # TimeSeries = np.zeros((172,85,246))
    # corr = np.zeros((172,246,246))
    # pcorr = np.zeros((172,246,246))
    
    ### ------------- origianl feature matrix 524*524 ----------------------###
    feature_matrix = feature_matrix.reshape((172,85,524))
    TimeSeries = np.zeros((172,85,524))
    corr = np.zeros((172,524,524))
    pcorr = np.zeros((172,524,524))
    
    TimeSeries = feature_matrix
    corr = subject_connectivity(TimeSeries, 'correlation')
    pcorr = subject_connectivity(TimeSeries, 'partial correlation')

    all_networks_corr = []
    all_networks_pcorr = []
    all_features = []
    for i in range(corr.shape[0]):
        # features = feature_matrix[i]
        matrix_corr = corr[i]
        matrix_pcorr = pcorr[i]

        all_networks_corr.append(matrix_corr)
        all_networks_pcorr.append(matrix_pcorr)
        # all_features.append(features)

    
    norm_networks_corr = [np.arctanh(mat) for mat in all_networks_corr] # 标准化的过程
    norm_networks_pcorr = [np.arctanh(mat) for mat in all_networks_pcorr] # 标准化的过程

    networks_corr = np.stack(norm_networks_corr)
    networks_pcorr = np.stack(norm_networks_pcorr)
    # networks_features = np.stack(all_features)

    if not os.path.exists('D:/ResearchGroup/ProjectCode/DMGPK/MyData/NC_AD/raw'):
        os.makedirs('D:/ResearchGroup/ProjectCode/DMGPK/MyData/NC_AD/raw')

    for i in range(networks_pcorr.shape[0]):
        dd.io.save(os.path.join('D:/ResearchGroup/ProjectCode/DMGPK/MyData/NC_AD/raw',str(i)+'.h5'),{'corr':networks_corr[i],'pcorr':networks_pcorr[i], 'label':labely[i]%5})
        print(i)

   # change the save paths and labely[i]%3