import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import h5py

from absl import app
from absl import flags

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

from del_qsar import models, featurizers, splitters, metrics, losses

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'DD1S_CAIX_QSAR.csv', 'csv filename')
flags.DEFINE_string('hyperparams', None, 'csv with hyperparameters of saved models')
flags.DEFINE_string('fps_h5', 'x_DD1S_CAIX_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_list('exp', ['exp_tot'], 'Column header(s) for data counts: experiment')
flags.DEFINE_list('beads', ['beads_tot'], 'Column header(s) for data counts: beads')
flags.DEFINE_list('splitter', ['random'], 'How molecules are split for training/testing')
flags.DEFINE_string('out', None, 'Experiment label (subfolder)')
flags.DEFINE_string('device', 'cuda:0', 'Device (set to cuda:0 if using GPU)')
flags.DEFINE_enum('random_type', 'shuffle_preds', ['shuffle_preds', 'predict_all_ones'], 'Type of random baseline')
flags.DEFINE_enum('eval_metric', 'NLL', ['NLL', 'MSE', 'rank_corr_coeff'], 'Metric for model evaluation')

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
        
    if 'DD1S_CAIX' in FLAGS.csv and 'cycle' not in FLAGS.splitter[0]:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'DD1S_CAIX_{FLAGS.random_type}_random_baseline_{FLAGS.eval_metric}_random_split.csv')
        split_name = 'random'
    elif 'DD1S_CAIX' in FLAGS.csv and 'cycle' in FLAGS.splitter[0]:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'DD1S_CAIX_{FLAGS.random_type}_random_baseline_{FLAGS.eval_metric}_cycle{"".join([c[-1:] for c in FLAGS.splitter])}_split.csv')
        split_name = f'cycle{"+".join([c[-1:] for c in FLAGS.splitter])}'
    elif 'sEH' in FLAGS.exp[0] and 'cycle' not in FLAGS.splitter[0]:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_sEH_{FLAGS.random_type}_random_baseline_{FLAGS.eval_metric}_random_split.csv')
        split_name = 'random'
    elif 'sEH' in FLAGS.exp[0] and 'cycle' in FLAGS.splitter[0]:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_sEH_{FLAGS.random_type}_random_baseline_{FLAGS.eval_metric}_cycle{"".join([c[-1:] for c in FLAGS.splitter])}_split.csv')
        split_name = f'cycle{"+".join([c[-1:] for c in FLAGS.splitter])}'
    elif 'SIRT2' in FLAGS.exp[0] and 'cycle' not in FLAGS.splitter[0]:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_SIRT2_{FLAGS.random_type}_random_baseline_{FLAGS.eval_metric}_random_split.csv')
        split_name = 'random'
    elif 'SIRT2' in FLAGS.exp[0] and 'cycle' in FLAGS.splitter[0]:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_SIRT2_{FLAGS.random_type}_random_baseline_{FLAGS.eval_metric}_cycle{"".join([c[-1:] for c in FLAGS.splitter])}_split.csv')
        split_name = f'cycle{"+".join([c[-1:] for c in FLAGS.splitter])}'
        
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
        # GPU?
        logging.info(f'CUDA available? {torch.cuda.is_available()}')
        lf.write(f'{datetime.now()} INFO: CUDA available? {torch.cuda.is_available()}\n')
        
    # Get data
    df_data = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments', 'datasets', FLAGS.csv))
    hyperparams = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.hyperparams))
    num_replicates = 5
    
    if 'CAIX' in FLAGS.csv:
        dataset_name = 'DD1S_CAIX'
    elif 'sEH' in FLAGS.exp[0]:
        dataset_name = 'triazine_sEH'
    elif 'SIRT2' in FLAGS.exp[0]:
        dataset_name = 'triazine_SIRT2'
    
    if 'triazine' in FLAGS.csv:
        for col in df_data.columns:
            if ' ' in col:
                df_data = df_data.rename(columns={col: col.replace(' ', '_')})
                
    # Extract counts and get calculated enrichments
    exp_counts = np.array(df_data[FLAGS.exp], dtype='int')
    bead_counts = np.array(df_data[FLAGS.beads], dtype='int')
    exp_tot = np.sum(exp_counts, axis=0) # column sums
    bead_tot = np.sum(bead_counts, axis=0)
    
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

    if FLAGS.splitter[0] == 'random':
        splitter = splitters.RandomSplitter()
    else:
        if len(FLAGS.splitter) == 1:
            splitter = splitters.OneCycleSplitter(FLAGS.splitter, LOG_FILE)
        elif len(FLAGS.splitter) == 2:
            splitter = splitters.TwoCycleSplitter(FLAGS.splitter, LOG_FILE)
        elif len(FLAGS.splitter) == 3:
            splitter = splitters.ThreeCycleSplitter(FLAGS.splitter, LOG_FILE)
        else:
            raise ValueError('Unknown splitter')
    
    all_avg_vals = []
    for seed in range(num_replicates):
        torch.manual_seed(seed)
        logging.info(f'Seed {seed}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: Seed {seed}\n')
            
        train_slice, valid_slice, test_slice = splitter(x, df_data, seed=seed)

        if FLAGS.random_type == 'shuffle_preds':
            df_row = hyperparams[
                hyperparams['dataset'].isin([dataset_name]) &
                hyperparams['model type'].isin(['FP-FFNN']) &
                hyperparams['seed'].isin([seed]) &
                hyperparams['split'].isin(['random']) 
            ]  
            layer_sizes = [int(s) for s in list(df_row['layer sizes'])[0].split(', ')]
            dropout = float(df_row['dropout'])
            input_size = x.shape[1]
            
            model = models.MLP(input_size, [int(size) for size in layer_sizes], dropout=dropout, torch_seed=seed)
            model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                          f'models/{dataset_name}/FP-FFNN', 
                                                          f'random_seed_{seed}.torch')))
            if FLAGS.device:
                model = model.to(FLAGS.device)
            test_enrichments = model.predict_on_x(
                x[test_slice, :], batch_size=1024,
                device=FLAGS.device,
            )
            np.random.seed(seed)
            np.random.shuffle(test_enrichments)
            
            logging.info(str(model))    
            with open(LOG_FILE, 'a') as lf:
                lf.write(f'{datetime.now()} INFO: ' + str(model) + '\n')
        else:
            test_enrichments = np.ones(len(test_slice))
        
        R = torch.FloatTensor(np.array([test_enrichments]).T)
        k1 = torch.FloatTensor(exp_counts[test_slice, :])
        k2 = torch.FloatTensor(bead_counts[test_slice, :])
        n1 = float(exp_tot)
        n2 = float(bead_tot)

        if FLAGS.eval_metric == 'NLL':
            metric_name = 'NLL loss'
            vals = losses.loss_fn_nlogprob(R, k1, k2, n1, n2)
            vals = np.array(vals, dtype=float)
            val = np.sum(vals, axis=0) / vals.shape[0]
            val = val[0]
            logging.info(f'Average NLL loss:    {"{0:.5f}".format(val)}')
            with open(LOG_FILE, 'a') as lf:      
                lf.write(f'{datetime.now()} Average NLL loss:    {"{0:.5f}".format(val)}\n')
            all_avg_vals.append(val)
        elif FLAGS.eval_metric == 'MSE':
            metric_name = 'MSE loss'
            vals = losses.loss_fn_MSE(R, k1, k2, n1, n2)
            vals = np.array(vals, dtype=float)
            val = np.sum(vals, axis=0) / vals.shape[0]
            val = val[0]
            logging.info(f'Average MSE loss:    {"{0:.5f}".format(val)}')
            with open(LOG_FILE, 'a') as lf:      
                lf.write(f'{datetime.now()} Average MSE loss:    {"{0:.5f}".format(val)}\n')
            all_avg_vals.append(val)
        else:
            metric_name = 'rank correlation coefficient'
            val = metrics.get_spearman_r(R, k1, k2, n1, n2)
            logging.info(f'Average rank correlation coefficient:    {"{0:.5f}".format(val)}')
            with open(LOG_FILE, 'a') as lf:      
                lf.write(f'{datetime.now()} Average rank correlation coefficient:    {"{0:.5f}".format(val)}\n')
            all_avg_vals.append(val)
                         
        # Record
        if not os.path.isfile(RESULTS_FILE):
            with open(RESULTS_FILE, 'w') as fid:
                cols = ['Time (of recording)']
                cols.append('Data split')
                cols.append('Random seed')
                cols.append(metric_name)
                fid.write('\t'.join(cols) + '\n')
        with open(RESULTS_FILE, 'a') as fid:
            cols = [str(datetime.now().now())]
            cols.append(split_name)
            cols.append(str(seed))
            cols.append(str(val))
            fid.write('\t'.join(cols) + '\n')
        fid.close()
    
    avg_across_seeds = sum(all_avg_vals) / len(all_avg_vals)
    logging.info(f'Average across seeds:    {"{0:.5f}".format(avg_across_seeds)}')
    with open(LOG_FILE, 'a') as lf:      
        lf.write(f'{datetime.now()} Average across seeds:    {"{0:.5f}".format(avg_across_seeds)}\n')
                     
if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
