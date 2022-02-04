from timeit import default_timer as time
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

from del_qsar import featurizers, splitters, models, losses, metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'triazine_lib_sEH_SIRT2_QSAR.csv', 'csv filename')
flags.DEFINE_string('fps_h5', 'x_triazine_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_string('exp', 'sEH_[strep]_tot', 'Column header for data counts: experiment')
flags.DEFINE_string('beads', 'beads-linker-only_[strep]_tot', 'Column header for data counts: beads')
flags.DEFINE_list('splitter', ['random'], 'How molecules are split for training/testing')
flags.DEFINE_integer('seed', 0, 'Random seed for data splitting')
flags.DEFINE_enum('featurizer', 'fingerprint', ['fingerprint', 'onehot'], 'How molecules are featurized')
flags.DEFINE_string('out', None, 'Experiment label (subfolder)')
flags.DEFINE_integer('n_trials', 1, 'Number of optuna trials to run')

if not os.path.isdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results')):
    os.mkdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results'))
    
dt = datetime.today()
DATE = os.path.join(DELQSAR_ROOT, 'experiments', 'results', 
                    f'{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}')
if not os.path.isdir(DATE):
    os.mkdir(DATE)

def main(argv):
    del argv
    
    start = time()  

    N_NEIGHBORS = 9

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
    if 'CAIX' in FLAGS.csv:
        dataset_label = 'DD1S_CAIX'
    elif 'sEH' in FLAGS.csv:
        dataset_label = 'triazine_sEH'
    elif 'SIRT2' in FLAGS.csv:
        dataset_label = 'triazine_SIRT2'

    logging.info(f'{len(df_data)} total compounds')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: {len(df_data)} total compounds\n')

    # Featurizer
    if FLAGS.featurizer == 'fingerprint':
        featurizer = featurizers.FingerprintFeaturizer()
        model_label = 'FP-KNN'
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
        model_label = 'OH-KNN'
        x = featurizer.prepare_x(df_data)
    else:
        raise ValueError('Unknown featurizer')
        
    INPUT_SIZE = x.shape[1] 
    logging.info(f'Input featurization is {INPUT_SIZE} long')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: Input featurization is {INPUT_SIZE} long\n')

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
        
        np.random.shuffle(test_slice)
        test_subset = test_slice[:int(0.1 * len(test_slice))]
    else:
        if len(FLAGS.splitter) == 1:
            splitter = splitters.OneCycleSplitter(FLAGS.splitter, LOG_FILE)
            splitter_name = FLAGS.splitter[0]
        elif len(FLAGS.splitter) == 2:
            splitter = splitters.TwoCycleSplitter(FLAGS.splitter, LOG_FILE)
            if 'cycle3' not in FLAGS.splitter:
                splitter_name = 'cycle12'
            elif 'cycle2' not in FLAGS.splitter:
                splitter_name = 'cycle13'
            else:
                splitter_name = 'cycle23'
        elif len(FLAGS.splitter) == 3:
            splitter = splitters.ThreeCycleSplitter(FLAGS.splitter, LOG_FILE)
            splitter_name = 'cycle123'
        else:
            raise ValueError('Unknown splitter')
        train_slice, valid_slice, test_slice = splitter(x, df_data, seed=FLAGS.seed)
        np.random.shuffle(test_slice)
        test_subset = test_slice[:int(0.1 * len(test_slice))]

    logging.info(f'Test subset length: {len(test_subset):12d}')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: Test subset length: {len(test_subset):12d}\n')

    # Load model
    saved_model = joblib.load(open(os.path.join(DELQSAR_ROOT, 'experiments', 'models', f'{dataset_label}', f'{model_label}', f'k_{N_NEIGHBORS}', f'{splitter_name}_seed_{FLAGS.seed}.joblib'), 'rb'))
    knn = models.kNN(task_type='regression', n_neighbors=N_NEIGHBORS, model=saved_model)

    logging.info('Model loaded')
    with open(LOG_FILE, 'a') as lf:
        lf.write('Model loaded\n')
    
    # Evaluations
    def evaluate_on_slice(eval_slice, slice_label='Test subset'):       
        test_enrichments = knn.predict_on_x(
            x[eval_slice, :],
        )

        logging.info(f'{slice_label} ({len(test_enrichments)} compounds): Predictions generated')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: {slice_label} ({len(test_enrichments)} compounds): Predictions generated\n')

        np.save(os.path.join(SAVE_ROOT, 'test_preds.npy'), test_enrichments)
        
        logging.info(f'{slice_label} ({len(test_enrichments)} compounds): Predictions saved')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: {slice_label} ({len(test_enrichments)} compounds): Predictions saved\n')
      
    # Evaluate on test set
    evaluate_on_slice(test_subset, slice_label='Test subset')
    
    total = time() - start
    m, s = divmod(total, 60)
    h, m = divmod(int(m), 60)
    logging.info(f'Total time elapsed: {h}h {m}m {s:0.2f}s\n')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: Total time elapsed: {h}h {m}m {s:0.2f}s\n\n')
    lf.close()

if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
    