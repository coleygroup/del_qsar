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

from del_qsar import featurizers, splitters, models

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'triazine_lib_sEH_SIRT2_QSAR.csv', 'csv filename')
flags.DEFINE_string('fps_h5', 'x_triazine_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_string('exp', 'sEH_[strep]_tot', 'Column header for data counts: experiment')
flags.DEFINE_string('beads', 'beads-linker-only_[strep]_tot', 'Column header for data counts: beads')
flags.DEFINE_enum('featurizer', 'fingerprint', ['fingerprint', 'onehot'], 'How molecules are featurized')
flags.DEFINE_list('splitter', ['random'], 'How molecules are featurized')
flags.DEFINE_integer('seed', 0, 'Random seed for data splitting')
flags.DEFINE_integer('n_neighbors', 9, 'Number of neighbors')
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
    if 'triazine' in FLAGS.csv:
        for col in df_data.columns:
            if ' ' in col:
                df_data = df_data.rename(columns={col: col.replace(' ', '_')})
    
    # Extract counts
    exp_counts = np.array(df_data[[FLAGS.exp]], dtype='int')
    bead_counts = np.array(df_data[[FLAGS.beads]], dtype='int')
    logging.info(f'{len(df_data)} total compounds')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: {len(df_data)} total compounds\n')

    # Featurizer
    if FLAGS.featurizer == 'fingerprint':
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
        
    elif FLAGS.featurizer == 'onehot':
        featurizer = featurizers.OneHotFeaturizer(df_data)
        x = featurizer.prepare_x(df_data)
        
    elif FLAGS.featurizer == 'graph':
        raise ValueError('Graph featurization not supported for kNN baseline')
        
    else:
        raise ValueError('Unknown featurizer')
        
    # Define different splits
    if FLAGS.splitter[0] == 'random':
        splitter = splitters.RandomSplitter()
        splitter_name = 'random'
        train_slice, valid_slice, test_slice = splitter(x, df_data, seed=FLAGS.seed)
        print('Random split:')
        print(f'Train: {train_slice}')
        print(f'Valid: {valid_slice}')
        print(f'Test: {test_slice}')
        with open(LOG_FILE, 'a') as lf:
            lf.write('\nRandom split:\n')
            lf.write(f'Train: {train_slice}\n')
            lf.write(f'Valid: {valid_slice}\n')
            lf.write(f'Test: {test_slice}\n\n')
    else:
        if len(FLAGS.splitter) == 1:
            splitter = splitters.OneCycleSplitter(FLAGS.splitter, LOG_FILE)
            splitter_name = FLAGS.splitter[0]
        elif len(FLAGS.splitter) == 2:
            splitter = splitters.TwoCycleSplitter(FLAGS.splitter, LOG_FILE)
            splitter_name = FLAGS.splitter[0] + FLAGS.splitter[1][-1]
        elif len(FLAGS.splitter) == 3:
            splitter = splitters.ThreeCycleSplitter(FLAGS.splitter, LOG_FILE)
            splitter_name = 'cycle123'
        else:
            raise ValueError('Unknown splitter')
        train_slice, valid_slice, test_slice = splitter(x, df_data, seed=FLAGS.seed)

    true_labels_R = None
    model = models.kNN(task_type='regression', n_neighbors=FLAGS.n_neighbors)

    try:
        model.train_on_del(x, exp_counts, bead_counts, train_slice, valid_slice, 
                           true_labels=true_labels_R, save_path=os.path.join(SAVE_ROOT, f'{splitter_name}_seed_{FLAGS.seed}'), log_path=LOG_FILE)
    except KeyboardInterrupt:
        logging.warning('Training interrupted by KeyboardInterrupt!')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} WARNING: Training interrupted by KeyboardInterrupt!\n')
            
    logging.info('Training completed')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: Training completed')
                         
if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
