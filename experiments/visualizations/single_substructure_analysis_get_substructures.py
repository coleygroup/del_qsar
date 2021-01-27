import os
import sys
import logging
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd
import rdkit.Chem as Chem
import hickle as hkl

from absl import app
from absl import flags

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

from del_qsar import featurizers

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'triazine_lib_sEH_SIRT2_QSAR.csv', 'csv filename')
flags.DEFINE_string('fps_h5', 'x_triazine_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_enum('dataset_label', 'triazine_sEH', ['DD1S_CAIX', 'triazine_sEH', 'triazine_SIRT2'], 'Dataset label')
flags.DEFINE_integer('seed', 0, 'Random seed for data splitting and weight initialization')
flags.DEFINE_list('bits_of_interest', None, 'Bits of interest (top 5 and bottom 3)')
    
def charge(a):
    """Returns a SMARTS substring describing the atomic charge."""
    if a.GetFormalCharge() >= 0:
        return f'+{a.GetFormalCharge()}'
    return f'-{abs(a.GetFormalCharge())}'

def getMorganFingerprintAtomSymbols(mol):
    """Generate custom atomSymbols based on the specificity of an atom
    definition used by Morgan Fingerprints. Namely,
        - atom ID
        - degree
        - number of Hs
        - ring membership
        - charge
    These are based on getConnectivityInvariants from FingerprintUtil.cpp at
    https://github.com/rdkit/rdkit/blob/75f03412ef151a4dc14dfee986e29c3690a4c071/Code/GraphMol/Fingerprints/FingerprintUtil.cpp#L254
    """
    atomSymbols = []
    for a in mol.GetAtoms():
        atomSymbols.append(
            f'[#{a.GetAtomicNum()};D{a.GetDegree()};H{a.GetTotalNumHs()};R{mol.GetRingInfo().NumAtomRings(a.GetIdx())};{charge(a)}]')
    return atomSymbols

def getFragmentForMolBit(smi, mol, mol_idx, atomSymbols, cpd_id, bit, info_all, submol_freq_distrib, 
                        smarts_to_smis, submol_to_cpd_indices, submol_to_bit, bits_to_draw):
    """Returns updated dictionaries and examples after searching for substructure(s) in the 
    specified molecule that set the specified bit.
    """
    molAdded = False
    examples = []
    for j, example in enumerate(info_all[mol_idx][bit]):
        atom = example[0]
        radius = example[1]
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
        atoms = set()
        for bidx in env:
            atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
        if atoms:
            submol_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), bondsToUse=env, 
                                            rootedAtAtom=atom, isomericSmiles=True, allBondsExplicit=True)
            submol_sm = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), atomSymbols=atomSymbols, 
                                            bondsToUse=env, isomericSmiles=True, allBondsExplicit=True)
            if submol_sm not in submol_freq_distrib[bit]:
                logging.info(f'cpd_id: {cpd_id}')
                logging.info(f'SMILES string: {smi}')
                logging.info(f'bit ID: {bit}')
                logging.info(f'(atom, radius): {(atom, radius)}')
                logging.info(f'molecular fragment (SMILES): {submol_smi}')
                logging.info(f'molecular fragment (SMARTS): {submol_sm}\n')
                submol_freq_distrib[bit][submol_sm] = 1 
                if submol_sm not in smarts_to_smis:
                    smarts_to_smis[submol_sm] = [submol_smi]
                else:
                    smarts_to_smis[submol_sm].append(submol_smi)   
                if submol_sm not in submol_to_bit:
                    submol_to_bit[submol_sm] = [bit]
                else:
                    submol_to_bit[submol_sm].append(bit)       
                examples.append(j)
                submol_to_cpd_indices[submol_sm] = [mol_idx]
                if not molAdded:
                    bits_to_draw.append((cpd_id, mol, bit, info_all[mol_idx]))
                    molAdded = True
            else:
                submol_freq_distrib[bit][submol_sm] += 1
                submol_to_cpd_indices[submol_sm].append(mol_idx)
        else:
            atom_smi = mol.GetAtomWithIdx(atom).GetSmarts()
            atom_sm = Chem.MolFragmentToSmiles(mol, 
                        atomsToUse=atom,
                        atomSymbols=atomSymbols,
                        isomericSmiles=True, 
                        allBondsExplicit=True)
            if atom_sm not in submol_freq_distrib[bit]:
                logging.info(f'cpd_id: {cpd_id}')
                logging.info(f'SMILES string: {smi}')
                logging.info(f'bit ID: {bit}')
                logging.info(f'(atom, radius): {(atom, radius)}')
                logging.info(f'atom: {atom_sm}\n')
                submol_freq_distrib[bit][atom_sm] = 1
                if atom_sm not in smarts_to_smis:
                    smarts_to_smis[atom_sm] = [atom_smi]
                else:
                    smarts_to_smis[atom_sm].append(atom_smi)
                if atom_sm not in submol_to_bit:
                    submol_to_bit[atom_sm] = [bit]
                else:
                    submol_to_bit[atom_sm].append(bit)
                examples.append(j)
                submol_to_cpd_indices[atom_sm] = [mol_idx]
                if not molAdded:
                    bits_to_draw.append((cpd_id, mol, bit, info_all[mol_idx]))
                    molAdded = True
            else:
                submol_freq_distrib[bit][atom_sm] += 1
                submol_to_cpd_indices[atom_sm].append(mol_idx)
    return submol_freq_distrib, smarts_to_smis, submol_to_cpd_indices, submol_to_bit, bits_to_draw, examples

def main(argv):
    del argv
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    df_data = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments', 'datasets', FLAGS.csv))
    
    if os.path.isfile(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.fps_h5)):
        hf = h5py.File(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.fps_h5))
        x = np.array(hf['all_fps'])
        hf.close()
    else:
        featurizer = featurizers.FingerprintFeaturizer()
        x = featurizer.prepare_x(df_data)
        
    bits_of_interest = [int(bit) for bit in FLAGS.bits_of_interest]
    bit_to_cpd_row_indices = {bit: list([idx for idx in np.squeeze(np.where(x[:,bit]==1))]) for bit in bits_of_interest}
    
    bits_to_draw = []
    submol_freq_distrib = {bit: {} for bit in bits_of_interest} # store frequency distribution of substructures that 
                                                                # set each bit
    smarts_to_smis = {}
    submol_to_cpd_indices = {} # mapping to indices in df_data_hasbit
    submol_to_bit = {}
    examples_all = {bit: {} for bit in bits_of_interest} # check if there's more than one distinct bit-setting 
                                                         # substructure in the same molecule
    
    for bit in tqdm(bits_of_interest):
        df_data_hasbit = df_data.iloc[bit_to_cpd_row_indices[bit]]
        smis = df_data_hasbit['smiles']
        featurizer = featurizers.FingerprintFeaturizer()
        _, info_all = featurizer.prepare_x(df_data_hasbit, bitInfo=True)
        for i, smi in enumerate(smis):
            mol = Chem.MolFromSmiles(smi)
            atomSymbols = getMorganFingerprintAtomSymbols(mol)
            cpd_id = int(df_data[df_data['smiles']==smi]['cpd_id'].to_numpy()[0])
            if bit not in info_all[i]:
                continue
            fragment_logs = getFragmentForMolBit(smi, mol, i, atomSymbols, cpd_id, bit, info_all, submol_freq_distrib, 
                            smarts_to_smis, submol_to_cpd_indices, submol_to_bit, bits_to_draw)
            submol_freq_distrib = fragment_logs[0]
            smarts_to_smis = fragment_logs[1]
            submol_to_cpd_indices = fragment_logs[2]
            submol_to_bit = fragment_logs[3]
            bits_to_draw = fragment_logs[4]
            examples = fragment_logs[5]
            if examples:
                examples_all[bit][i] = examples

    hkl.dump(submol_freq_distrib, 
             os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 
                          f'submol_freq_distrib_{FLAGS.dataset_label}_FP-FFNN_random_seed_{FLAGS.seed}.hkl'), mode='w')
    hkl.dump(smarts_to_smis, 
             os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 
                          f'smarts_to_smis_{FLAGS.dataset_label}_FP-FFNN_random_seed_{FLAGS.seed}.hkl'), mode='w')
    hkl.dump(submol_to_cpd_indices, 
             os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 
                          f'submol_to_cpd_indices_{FLAGS.dataset_label}_FP-FFNN_random_seed_{FLAGS.seed}.hkl'), mode='w')
    hkl.dump(submol_to_bit, 
             os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 
                          f'submol_to_bit_{FLAGS.dataset_label}_FP-FFNN_random_seed_{FLAGS.seed}.hkl'), mode='w')
    hkl.dump(bits_to_draw, 
             os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 
                          f'bits_to_draw_{FLAGS.dataset_label}_FP-FFNN_random_seed_{FLAGS.seed}.hkl'), mode='w')
    hkl.dump(examples_all, 
             os.path.join(DELQSAR_ROOT, 'experiments', 'visualizations', 
                          f'examples_all_{FLAGS.dataset_label}_FP-FFNN_random_seed_{FLAGS.seed}.hkl'), mode='w')
    
if __name__ == '__main__':
    app.run(main)
