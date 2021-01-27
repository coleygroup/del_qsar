import os, sys
import numpy as np
from tqdm import tqdm
import h5py
from rdkit.Chem import AllChem
import rdkit.Chem as Chem

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

smiles = np.load(os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 'pubchem_smiles.npy'))
x = np.zeros((len(smiles), 4096), dtype=np.bool)
ctr = 0
for i, smi in enumerate(tqdm(smiles)):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
    if mol:
        ctr += 1
        x[i, :] = np.array(AllChem.GetMorganFingerprintAsBitVect(
                        mol,
                        radius=3,
                        nBits=4096,
                        useChirality=True,
                    ), dtype='bool')  
print(f'Number of molecules converted from SMILES: {ctr}')
try:
    hf = h5py.File('pubchem_fps_4096_bits.h5', 'w')
    hf.create_dataset('all_fps', data=x)
    hf.close()
    print(f'All fingerprints generated and stored in pubchem_fps_4096_bits')
except Exception as e:
    print(str(e))
    