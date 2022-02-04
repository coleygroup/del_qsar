from timeit import default_timer as time
import os
import sys
import logging
from datetime import datetime
import optuna
import pandas as pd
import numpy as np
import torch

from absl import app
from absl import flags

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

RESULTS_FILE = os.path.join(DELQSAR_ROOT, 'experiments', 'all_results.csv')

from del_qsar import splitters, losses, metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'triazine_lib_sEH_SIRT2_QSAR.csv', 'csv filename')
flags.DEFINE_string('exp', 'sEH_[strep]_tot', 'Column header for data counts: experiment')
flags.DEFINE_string('beads', 'beads-linker-only_[strep]_tot', 'Column header for data counts: beads')
flags.DEFINE_string('test_preds_npy', None, 'Path to saved test predictions (from del_qsar/experiments/results folder)')
flags.DEFINE_list('splitter', ['random'], 'How molecules are split for training/testing')
flags.DEFINE_enum('eval_metric', 'NLL', ['NLL', 'MSE', 'rank_corr_coeff'], 'Metric for evaluating models')
flags.DEFINE_string('out', None, 'Experiment label (subfolder)')
flags.DEFINE_integer('n_trials', 1, 'Number of optuna trials to run')

if not os.path.isdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results')):
    os.mkdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results'))
    
dt = datetime.today()
DATE = os.path.join(DELQSAR_ROOT, 'experiments', 'results', 
                    f'{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}')
if not os.path.isdir(DATE):
    os.mkdir(DATE)

def objective(trial):
    start = time()
    SAVE_ROOT = os.path.join(DATE, FLAGS.out)
    if not os.path.isdir(SAVE_ROOT):
        os.mkdir(SAVE_ROOT)
        
    SAVE_SUBFOLDER = os.path.join(SAVE_ROOT, '1_log_files')
    if not os.path.isdir(SAVE_SUBFOLDER):
        os.mkdir(SAVE_SUBFOLDER)
        
    LOG_FILE = os.path.join(SAVE_SUBFOLDER, 'run.log')
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    with open(LOG_FILE, 'a') as lf:
        logging.info(f'Trial #{trial.number}')
        lf.write(f'{datetime.now()} INFO: Trial #{trial.number}\n')
        logging.info(f'Parameters: {trial.params}')
        lf.write(f'{datetime.now()} INFO: Parameters: {trial.params}\n\n')
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
    exp_tot = np.sum(exp_counts, axis=0) # column sums
    bead_tot = np.sum(bead_counts, axis=0)
    logging.info(f'{len(df_data)} total compounds')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: {len(df_data)} total compounds\n')

    # Define different splits
    if FLAGS.splitter[0] == 'random':
        splitter = splitters.RandomSplitter()
        train_slice, valid_slice, test_slice = splitter(None, df_data, seed=0)
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
        elif len(FLAGS.splitter) == 2:
            splitter = splitters.TwoCycleSplitter(FLAGS.splitter, LOG_FILE)
        elif len(FLAGS.splitter) == 3:
            splitter = splitters.ThreeCycleSplitter(FLAGS.splitter, LOG_FILE)
        else:
            raise ValueError('Unknown splitter')
        train_slice, valid_slice, test_slice = splitter(None, df_data, seed=0)
        
    np.random.shuffle(test_slice)
    test_subset = test_slice[:int(0.1 * len(test_slice))]

    logging.info(f'Test subset length: {len(test_subset):12d}')

    test_preds = np.load(os.path.join(DELQSAR_ROOT, 'experiments', 'results', FLAGS.test_preds_npy))
    test_preds = torch.FloatTensor(test_preds)
    
    # Evaluations
    k1 = torch.FloatTensor(exp_counts[test_subset])
    k2 = torch.FloatTensor(bead_counts[test_subset])
    n1 = float(exp_tot)
    n2 = float(bead_tot)
    
    if FLAGS.eval_metric == 'NLL':
        test_losses = losses.loss_fn_nlogprob(test_preds, k1, k2, n1, n2)
        test_losses = np.array(test_losses, dtype=float)
        avg_test_loss = np.sum(test_losses, axis=0) / test_losses.shape[0]
        formatted_avg_test_loss = '{0:.5f}'.format(np.squeeze(avg_test_loss))
        logging.info(f'Average NLL loss: {formatted_avg_test_loss}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: Average NLL loss: {formatted_avg_test_loss}\n')
    elif FLAGS.eval_metric == 'MSE':
        test_losses = losses.loss_fn_MSE(test_preds, k1, k2, n1, n2)
        test_losses = np.array(test_losses, dtype=float)
        avg_test_loss = np.sum(test_losses, axis=0) / test_losses.shape[0]
        formatted_avg_test_loss = '{0:.5f}'.format(np.squeeze(avg_test_loss))
        logging.info(f'Average MSE loss: {formatted_avg_test_loss}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: Average MSE loss: {formatted_avg_test_loss}\n')
    elif FLAGS.eval_metric == 'rank_corr_coeff':
        rank_corr_coeff = metrics.get_spearman_r(test_preds, k1, k2, n1, n2)
        formatted_rank_corr_coeff = '{0:.5f}'.format(rank_corr_coeff)
        logging.info(f'Rank correlation coefficient: {formatted_rank_corr_coeff}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: Rank correlation coefficient: {formatted_rank_corr_coeff}\n')

    # Record to results file
    if not os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as fid:
            cols = ['Time (of recording)']
            for f in FLAGS.get_key_flags_for_module(sys.argv[0]):
                flag,value = f.serialize().split('=')
                cols.append(flag[2:])
            cols.append('Training losses')
            cols.append('Training rank corr coeff')
            cols.append('Validation losses')
            cols.append('Validation rank corr coeff')
            cols.append('Valid % predicted R in (R_lb, R_ub)')   
            cols.append('Valid ROC AUC')
            cols.append('Valid PR AUC')
            cols.append('Test loss')
            cols.append('Test rank corr coeff')
            cols.append('Test % predicted R in (R_lb, R_ub)')         
            cols.append('Test ROC AUC')
            cols.append('Test PR AUC')
            fid.write('\t'.join(cols) + '\n')
    with open(RESULTS_FILE, 'a') as fid:
        cols = [str(datetime.now().now())]
        for f in FLAGS.get_key_flags_for_module(sys.argv[0]):
            flag,value = f.serialize().split('=')
            cols.append(value)
        cols.append('')
        cols.append('')
        cols.append('')
        cols.append('')
        cols.append('')
        cols.append('')
        cols.append('')
        if FLAGS.eval_metric in ['NLL', 'MSE']:
            cols.append(str(np.squeeze(avg_test_loss)))
            cols.append('')
            cols.append('')
            cols.append('')
            cols.append('')
        else: # rank corr coeff
            cols.append('')
            cols.append(str(rank_corr_coeff))
            cols.append('')
            cols.append('')
            cols.append('')
        fid.write('\t'.join(cols) + '\n')
    fid.close()
    
    total = time() - start
    m, s = divmod(total, 60)
    h, m = divmod(int(m), 60)
    logging.info(f'Total time for trial #{trial.number}: {h}h {m}m {s:0.2f}s\n')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: Total time for trial #{trial.number}: {h}h {m}m {s:0.2f}s\n\n')
    lf.close()
    
    if FLAGS.eval_metric == 'rank_corr_coeff':
        return rank_corr_coeff
    else:
        return avg_test_loss
  
def main(argv):
    del argv
    
    # Using code from https://stackoverflow.com/a/58833843 to keep track of number of unique trials
    if FLAGS.eval_metric != 'rank_corr_coeff':
        study = optuna.create_study()
    else:
        study = optuna.create_study(direction='maximize')
    unique_trials = FLAGS.n_trials
    while unique_trials > len(set(str(t.params) for t in study.trials)):
        study.optimize(objective, n_trials=1)

    SAVE_ROOT = os.path.join(DATE, FLAGS.out)
    if not os.path.isdir(SAVE_ROOT):
        os.mkdir(SAVE_ROOT)
        
    SAVE_SUBFOLDER = os.path.join(SAVE_ROOT, '1_log_files')
    if not os.path.isdir(SAVE_SUBFOLDER):
        os.mkdir(SAVE_SUBFOLDER)
        
    LOG_FILE = os.path.join(SAVE_SUBFOLDER, 'run.log')
    TRIALS_LOG = os.path.join(SAVE_SUBFOLDER, 'trials.csv')
    
    trial = study.best_trial
    with open(LOG_FILE, 'a') as lf:
        print('Number of trials: {}'.format(len(study.trials)))
        lf.write(f'\n{datetime.now()} INFO: ' + 'Number of trials: {}'.format(len(study.trials)) + '\n')
        
        print(f'Best trial: Trial #{trial.number}')
        lf.write(f'{datetime.now()} INFO: Best trial: Trial #{trial.number}\n')        
        
        print('  Value: {}'.format(trial.value))
        lf.write(f'{datetime.now()} INFO: ' + '  Value: {}'.format(trial.value) + '\n')  
        
        print('  Params: ')
        lf.write(f'{datetime.now()} INFO: ' + '  Params: \n')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
            lf.write(f'{datetime.now()} INFO: ' + '    {}: {}'.format(key, value) + '\n')
    lf.close()
    
    study.trials_dataframe().to_csv(TRIALS_LOG, sep='\t', index=False)
    
if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
