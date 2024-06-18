# Deep Multimodal Interpretable Graph Pooling Network with Knowledge Learning for Alzheimer's Disease Diagnosis

A preliminary implementation of DMGPK.

## Usage
### Setup
**pip**

See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```
**PYG**

To install pyg library, [please refer to the document](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

All fMRI data and SNP data are downloaded from ADNI database.
The fMRI data and SNP data are preprocessed by the DPABI toolbox and Plink software, respectively.

# SNP Preprocess and Encoding Representation. 
```
python SNPProcess.py
```
# The fMRI data and gene data of each sample are used to construct a heterogeneous graph. 

How to construct the graphs?
```
python Mydataprocess.py
```

# Position Encoding of demographic Knowledge. 
```
python position_embedding.py
```

### How to run classification?
Training and testing are integrated in file `main.py`. To run
```
python main.py 
```
