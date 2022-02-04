import os
import sys
import logging
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import h5py

from absl import app
from absl import flags

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

from del_qsar import featurizers

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', None, 'path (from the experiments folder) to saved FP-KNN trained on the entire DD1S CAIX dataset')

if not os.path.isdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results')):
    os.mkdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results'))
    
dt = datetime.today()
DATE = os.path.join(DELQSAR_ROOT, 'experiments', 'results', 
                    f'{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}')
if not os.path.isdir(DATE):
    os.mkdir(DATE)

def main(argv):
    del argv
    
    SAVE_ROOT = os.path.join(DATE, 'outliers_nearest_neighbors')
    if not os.path.isdir(SAVE_ROOT):
        os.mkdir(SAVE_ROOT)
        
    LOG_FILE = os.path.join(SAVE_ROOT, 'run.log')
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Get data
    df_data = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments/datasets/DD1S_CAIX_QSAR.csv'))
    logging.info(f'{len(df_data)} total compounds')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: {len(df_data)} total compounds\n')
        
    featurizer = featurizers.FingerprintFeaturizer()
    if os.path.isfile(os.path.join(DELQSAR_ROOT, 'experiments', 'x_DD1S_CAIX_2048_bits_all_fps.h5')):
        hf = h5py.File(os.path.join(DELQSAR_ROOT, 'experiments', 'x_DD1S_CAIX_2048_bits_all_fps.h5'))
        x = np.array(hf['all_fps'])
        hf.close()
        logging.warning('Loaded fingerprints from x_DD1S_CAIX_2048_bits_all_fps.h5')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} WARNING: Loaded fingerprints from x_DD1S_CAIX_2048_bits_all_fps.h5\n')
    else:
        logging.info('Generating fingerprints')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: Generating fingerprints\n')
        x = featurizer.prepare_x(df_data)
        
    INPUT_SIZE = x.shape[1] 
    logging.info(f'Input featurization is {INPUT_SIZE} long')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: Input featurization is {INPUT_SIZE} long\n')
    
    np.random.seed(0)
    data_indices = np.arange(int(len(df_data)))
    np.random.shuffle(data_indices)
    np.random.shuffle(data_indices)
            
    saved_model = joblib.load(open(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.model_path), 'rb'))
    
    logging.info('Model loaded')
    with open(LOG_FILE, 'a') as lf:
        lf.write('Model loaded\n')
    
    outlier_fps = x[[3857, 87394, 81920, 104264, 66578]]
        
    for i, fp in enumerate(outlier_fps):
        nn_dist, nn_idx = saved_model.kneighbors(X=[fp], n_neighbors=2)
        nn_dist = np.squeeze(nn_dist)[1]
        nn_idx = np.squeeze(nn_idx)[1]
        nn_df_data_idx = data_indices[nn_idx]
        nn_smi = df_data.iloc[nn_df_data_idx]['smiles']
        nn_info = df_data.iloc[nn_df_data_idx]
        print(f'\nOutlier #{i}')
        print('Nearest neighbor:')
        print(f'shuffled idx:    {nn_idx}')
        print(f'df_data idx:     {nn_df_data_idx}')
        print(f'Distance:        {nn_dist}')
        print(f'SMILES:          {nn_smi}')
        print(str(nn_info))
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'\n\nOutlier #{i}\n')
            lf.write('Nearest neighbor:\n')
            lf.write(f'shuffled idx:    {nn_idx}\n')
            lf.write(f'df_data idx:     {nn_df_data_idx}\n')
            lf.write(f'Distance:        {nn_dist}\n')
            lf.write(f'SMILES:          {nn_smi}\n')
            lf.write(str(nn_info))
    
if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
