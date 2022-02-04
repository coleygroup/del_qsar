import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import h5py

from absl import app
from absl import flags

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

RESULTS_FILE = os.path.join(DELQSAR_ROOT, 'experiments', 'all_results.csv')

from del_qsar import featurizers, models

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'DD1S_CAIX_QSAR.csv', 'csv filename')
flags.DEFINE_string('fps_h5', 'x_DD1S_CAIX_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_string('exp', 'exp_tot', 'Column header for data counts: experiment')
flags.DEFINE_string('beads', 'beads_tot', 'Column header for data counts: beads')
flags.DEFINE_string('out', None, 'Experiment label (subfolder)')

if not os.path.isdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results')):
    os.mkdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results'))
    
dt = datetime.today()
DATE = os.path.join(DELQSAR_ROOT, 'experiments', 'results', 
                    f'{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}')
if not os.path.isdir(DATE):
    os.mkdir(DATE)

def main(argv):
    del argv
    
    SAVE_ROOT = os.path.join(DATE, FLAGS.out)
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
    with open(LOG_FILE, 'a') as lf:
        logging.info('FLAGS:')
        lf.write(f'{datetime.now()} INFO: FLAGS:\n')
        for f in FLAGS.get_key_flags_for_module(sys.argv[0]):
            logging.info(f.serialize())
            lf.write(f'{datetime.now()} INFO: ' + f.serialize() + '\n')

    # Get data
    df_data = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments', 'datasets', FLAGS.csv))
    
    # Extract counts
    exp_counts = np.array(df_data[[FLAGS.exp]], dtype='int')
    bead_counts = np.array(df_data[[FLAGS.beads]], dtype='int')
    logging.info(f'{len(df_data)} total compounds')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: {len(df_data)} total compounds\n')

    # Featurizer
    featurizer = featurizers.FingerprintFeaturizer()
    if os.path.isfile(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.fps_h5)):
        hf = h5py.File(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.fps_h5))
        x = np.array(hf['all_fps'])
        hf.close()
        logging.warning(f'Loaded fingerprints from {FLAGS.fps_h5}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} WARNING: Loaded fingerprints from {FLAGS.fps_h5}\n')
    else:
        logging.info('Generating fingerprints')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: Generating fingerprints\n')
        x = featurizer.prepare_x(df_data)      
    
    true_labels_R = None
    model = models.kNN(task_type='regression', n_neighbors=9)
    
    np.random.seed(0)
    all_data = np.arange(int(len(df_data)))
    np.random.shuffle(all_data)
        
    try:
        model.train_on_del(x, exp_counts, bead_counts, all_data, None, 
                           true_labels=true_labels_R, save_path=os.path.join(SAVE_ROOT, 'DD1S_CAIX_all'))
    except KeyboardInterrupt:
        logging.warning('Training interrupted by KeyboardInterrupt!')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} WARNING: Training interrupted by KeyboardInterrupt!\n')            
                         
if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
