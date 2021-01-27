import functools
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import AllChem
import rdkit.Chem as Chem
import numpy as np
from tqdm import tqdm
from chemprop.data.data import MoleculeDatapoint

FP_SIZE = 2048
FP_RADIUS = 3

class Featurizer(object):
    pass


class GraphFeaturizer(Featurizer):
    def __init__(self, smis, targets):
        self.smiles = smis
        self.targets = targets
        
    def prepare_x(self):
        return [MoleculeDatapoint(
                smiles=smi,
                targets=[float(i) for i in targ]
               ) for smi, targ in zip(self.smiles, self.targets)]

    
class FingerprintFeaturizer(Featurizer):
    def __init__(self, fp_size=FP_SIZE, radius=FP_RADIUS):
        self.fp_size = fp_size
        self.radius = radius
        self.simmap_featurizer = functools.partial(
            SimilarityMaps.GetMorganFingerprint,
            radius=self.radius,
            nBits=self.fp_size,
            useChirality=True
        )

    def calc_fp(self, smi, bitInfo=False):
        if bitInfo:
            mol = Chem.MolFromSmiles(smi)
            info = {}
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.radius,
                    nBits=self.fp_size,
                    useChirality=True,
                    bitInfo=info,
                ), dtype='bool')        
            return fp, info
        else:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smi),
                    radius=self.radius,
                    nBits=self.fp_size,
                    useChirality=True,
                ), dtype='bool')

    def prepare_x(self, df_data, bitInfo=False):
        x = np.zeros((len(df_data), self.fp_size), dtype=np.bool)
        if bitInfo: 
            info_all = []
            for i, smi in enumerate(tqdm(df_data['smiles'])):
                fp, info = self.calc_fp(smi, bitInfo=True)
                x[i, :] = fp
                info_all.append(info)
            return x, info_all
        else:
            for i, smi in enumerate(tqdm(df_data['smiles'])):
                x[i, :] = self.calc_fp(smi, bitInfo=False)              
            return x


class OneHotFeaturizer(Featurizer):
    '''cycle_ids are 1-indexed'''
    tags = ['library_id', 'cycle1', 'cycle2', 'cycle3']

    def __init__(self, df_data):
        self.offsets = {}
        current_offset = -1
        for lib_id, df_data_lib in df_data[self.tags].groupby('library_id'):
            for cycnum in [1, 2, 3]:
                self.offsets[(lib_id, cycnum)] = current_offset
                current_offset += len(df_data_lib[f'cycle{cycnum}'].unique())
        self.length = current_offset + 1

    def prepare_x(self, df_data):
        x = np.zeros((len(df_data), self.length), dtype=np.bool)
        for i, indexed_row in enumerate(tqdm(df_data[self.tags].iterrows())):
            _, (lib_id, cyc1, cyc2, cyc3) = indexed_row
            x[i, self.offsets[lib_id, 1] + cyc1] = 1
            x[i, self.offsets[lib_id, 2] + cyc2] = 1
            x[i, self.offsets[lib_id, 3] + cyc3] = 1
        return x
    