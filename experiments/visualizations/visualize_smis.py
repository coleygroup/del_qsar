import os
import sys
from datetime import datetime
import logging
from collections import OrderedDict
import io
import numpy as np
import torch
import h5py
import pandas as pd
from PIL import Image
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

from absl import app
from absl import flags

def save_png(data, out_path):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    orig = [float(d) for d in img.size]
    scale = 300.0/72.0
    img.thumbnail([round(scale * d) for d in orig], Image.ANTIALIAS)
    img.save(out_path, 'PNG', dpi=(300.0, 300.0))

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

from del_qsar import featurizers, models

if not os.path.isdir(os.path.join(DELQSAR_ROOT, 'results')):
    os.mkdir(os.path.join(DELQSAR_ROOT, 'results'))

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'triazine_lib_sEH_SIRT2_QSAR.csv', 'csv filename')
flags.DEFINE_string('fps_h5', 'x_triazine_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_string('out', None, 'Output folder name')

flags.DEFINE_string('model_path', None, 'File path from experiments folder for saved model (should be a .torch file)')
flags.DEFINE_float('dropout', None, 'Dropout rate')
flags.DEFINE_list('layer_sizes', None, 'FFN layer sizes')
flags.DEFINE_list('cpd_ids', None, 'cpd_ids of the compounds to visualize with atom annotations')

def main(argv):
    del argv
    
    dt = datetime.today()
    
    DATE = os.path.join(DELQSAR_ROOT, 'experiments', 'results', 
                        f'{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}')
    if not os.path.isdir(DATE):
        os.mkdir(DATE)
    SAVE_ROOT = os.path.join(DATE, FLAGS.out)
    if not os.path.isdir(SAVE_ROOT):
        os.mkdir(SAVE_ROOT)
        
    LOG_FILE = os.path.join(SAVE_ROOT, 'run.log')
    with open(LOG_FILE, 'a') as lf:
        logging.info('FLAGS:')
        lf.write(f'{datetime.now()} INFO: FLAGS:\n')
        for f in FLAGS.get_key_flags_for_module(sys.argv[0]):
            logging.info(f.serialize())
            lf.write(f'{datetime.now()} INFO: ' + f.serialize() + '\n')
            
    df_data = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments', 'datasets', FLAGS.csv))
    
    smis, smis_to_cpd_ids, x_rows_visualize = [], {}, []
    for ID in FLAGS.cpd_ids:
        smi = df_data.iloc[int(ID)-1]['smiles'] 
        smis.append(smi)
        smis_to_cpd_ids[smi] = ID
        x_rows_visualize.append(int(ID)-1)
    
    featurizer = featurizers.FingerprintFeaturizer()  
    df_visualize = pd.DataFrame.from_dict({'smiles': smis})
    if os.path.isfile(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.fps_h5)):
        hf = h5py.File(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.fps_h5))
        x = np.array(hf['all_fps'])
        x_visualize = x[x_rows_visualize, :]
        hf.close()
    else:
        x_visualize = featurizer.prepare_x(df_visualize)
        
    input_size = x_visualize.shape[1]     
    print(f'Input featurization is {input_size} long')
    model = models.MLP(input_size, [int(size) for size in FLAGS.layer_sizes],
                dropout=FLAGS.dropout)
        
    model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.model_path)))
    model = model.to('cuda:0')
    print(model)
    
    model.eval()
    enrichments = model.predict_on_x(x_visualize, device='cuda:0')
    
    drawings = OrderedDict()

    # Get all weights
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi) 
        weights = SimilarityMaps.GetAtomicWeightsForModel(
            mol,
            featurizer.simmap_featurizer,
            lambda fp: model.predict_on_x(np.array(fp), device='cuda:0'), 
        )
        drawings[smi] = (mol,weights,enrichments[i])

    # Normalize
    scale = max(np.abs(w) for w in weights for _,weights,_ in drawings.values())

    # Show
    for smi, (mol,weights,enrichment) in drawings.items():
        logging.info(f'cpd_id: {smis_to_cpd_ids[smi]}')
        logging.info(f'SMILES: {smi}')
        logging.info(f'Predicted enrichment: {enrichment:.2f}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'cpd_id: {smis_to_cpd_ids[smi]}\n')
            lf.write(f'SMILES: {smi}\n')
            lf.write(f'Predicted enrichment: {enrichment:.2f}\n\n')
        lf.close()
        d = Draw.MolDraw2DCairo(600, 600)
        if scale != 0:
            SimilarityMaps.GetSimilarityMapFromWeights(mol, [w/scale for w in weights], draw2d=d)
        else:
            SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, draw2d=d)
        d.FinishDrawing()
        save_png(d.GetDrawingText(), os.path.join(SAVE_ROOT, 'cpd_id_' + str(smis_to_cpd_ids[smi]) + '.png'))
        
if __name__ == '__main__':
    app.run(main)
