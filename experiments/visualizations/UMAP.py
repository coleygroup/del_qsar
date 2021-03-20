import os
import sys
import h5py
import numba
import numpy as np
import umap
from absl import app
from absl import flags

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_threads', 40, 'Number of threads')
flags.DEFINE_string('pubchem_fps_h5', 'pubchem_fps_4096_bits.h5', 'HDF5 file with stored fingerprints for PubChem')
flags.DEFINE_string('DD1S_fps_h5', 'DOS-DEL-1_4096_bits_all_fps.h5', 'HDF5 file with stored fingerprints for DOS-DEL-1')
flags.DEFINE_string('triazine_fps_h5', 'triazine_4096_bits_all_fps.h5', 
                    'HDF5 file with stored fingerprints for the triazine library')

def main(argv):
    del argv
    np.random.seed(0)
    
    hf_pubchem = h5py.File(os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', FLAGS.pubchem_fps_h5))
    x_pubchem = np.array(hf_pubchem['all_fps'])
    hf_pubchem.close()
    
    hf_DD1S = h5py.File(os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', FLAGS.DD1S_fps_h5))
    x_DD1S = np.array(hf_DD1S['all_fps'])
    hf_DD1S.close()
    
    hf_triazine = h5py.File(os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', FLAGS.triazine_fps_h5))
    x_triazine = np.array(hf_triazine['all_fps'])
    hf_triazine.close()
    
    data_indices_DD1S = np.arange(int(x_DD1S.shape[0]))
    np.random.shuffle(data_indices_DD1S)
    train_subset_DD1S = data_indices_DD1S[:int(0.1*x_DD1S.shape[0])]
    x_DD1S_subset = x_DD1S[train_subset_DD1S,:]
    
    data_indices_triazine = np.arange(int(x_triazine.shape[0]))
    np.random.shuffle(data_indices_triazine)
    train_subset_triazine = data_indices_triazine[:int(0.1*x_triazine.shape[0])]
    x_triazine_subset = x_triazine[train_subset_triazine,:]

    x_all = np.concatenate((x_pubchem, x_DD1S_subset, x_triazine_subset), axis=0)
    print(f'Combined dataset shape: {x_all.shape}\n')
    data_indices_all = np.arange(int(x_all.shape[0]))
    np.random.shuffle(data_indices_all)
    x_all = x_all[data_indices_all,:]
    
    numba.set_num_threads(FLAGS.num_threads)
    r = umap.UMAP(metric='jaccard', verbose=True, low_memory=True)
    
    r.fit(x_all)
    umap_pubchem = r.transform(x_pubchem)
    umap_DD1S = r.transform(x_DD1S)
    umap_triazine = r.transform(x_triazine_subset)
    print(f'PubChem embedding shape: {umap_pubchem.shape}')
    print(f'DD1S embedding shape: {umap_DD1S.shape}')
    print(f'Triazine library embedding shape: {umap_triazine.shape}')
    np.save(os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 'pubchem_4096_bits_umap.npy'), umap_pubchem)
    np.save(os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 'DOS_DEL_1_4096_bits_umap.npy'), umap_DD1S)
    np.save(os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 'triazine_4096_bits_umap.npy'), umap_triazine)

if __name__ == '__main__':
    app.run(main)
    