import numpy as np
import scipy.io as sio
import h5py
from sklearn.preprocessing import MinMaxScaler

# SNP Encoding Representation Module

def SNP_encoder(X_SNP_tr):
    # Based on population, this encoder transforms the discrete SNP vectors to be numerical.
    # The encoder is fit by the training SNP data and applied to the testing SNP data.

    # Fit the encoding table
    encoder = np.empty(shape=(3, X_SNP_tr.shape[1]))
    for i in range(X_SNP_tr.shape[1]):
        for j in [0, 1, 2]:
            encoder[j, i] = np.array(X_SNP_tr[:, i] == j).sum() #将每一列的0 1 2的个数对应在0 1 2行

    encoder /= X_SNP_tr.shape[0]  # (3, 1275)

    X_E_SNP_tr = np.empty(shape=X_SNP_tr.shape)

    # Map the SNP values
    for sbj in range(X_SNP_tr.shape[0]):
        for dna in range(X_SNP_tr.shape[-1]):

            X_E_SNP_tr[sbj, dna] = encoder[..., dna][int(X_SNP_tr[sbj, dna])]

    # for sbj in range(X_SNP_ts.shape[0]):
    #     for dna in range(X_SNP_ts.shape[-1]):
    #         X_E_SNP_ts[sbj, dna] = encoder[..., dna][int(X_SNP_ts[sbj, dna])]

    return X_E_SNP_tr


if __name__ == '__main__':
    
    Gene = np.transpose(h5py.File('D:/ResearchGroup/ProjectCode/DMGPK/MyData/Gene/Gene_New.mat', 'r')['Gene_New'][()])
    Gene_map = np.zeros((358,278,85)) #一共358个样本
    for i in range(Gene.shape[0]):
        X_SNP_tr = Gene[i]
        Gene_map[i] = SNP_encoder(X_SNP_tr)

    sio.savemat('Gene_map.mat', {'Gene_map': Gene_map})


