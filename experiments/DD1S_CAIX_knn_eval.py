from timeit import default_timer as time
import os
import sys
import logging
from datetime import datetime
import joblib
import optuna
import pandas as pd
import numpy as np
import torch
import h5py

from absl import app
from absl import flags

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

RESULTS_FILE = os.path.join(DELQSAR_ROOT, 'experiments', 'all_results.csv')

from del_qsar import featurizers, splitters, models, losses, metrics
from del_qsar.enrichments import R_ranges

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'DD1S_CAIX_QSAR.csv', 'csv filename')
flags.DEFINE_string('fps_h5', 'x_DD1S_CAIX_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_string('exp', 'exp_tot', 'Column header for data counts: experiment')
flags.DEFINE_string('beads', 'beads_tot', 'Column header for data counts: beads')
flags.DEFINE_list('splitter', ['random'], 'How molecules are split for training/testing')
flags.DEFINE_integer('seed', 0, 'Random seed for data splitting')
flags.DEFINE_enum('eval_metric', 'NLL', ['NLL', 'MSE', 'rank_corr_coeff'], 'Metric for evaluating models')
flags.DEFINE_enum('featurizer', 'fingerprint', ['fingerprint', 'onehot'], 'How molecules are featurized')
flags.DEFINE_string('out', None, 'Experiment label (subfolder)')
flags.DEFINE_integer('n_trials', 5, 'Number of optuna trials to run')

if not os.path.isdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results')):
    os.mkdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results'))
    
dt = datetime.today()
DATE = os.path.join(DELQSAR_ROOT, 'experiments', 'results', 
                    f'{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}')
if not os.path.isdir(DATE):
    os.mkdir(DATE)

def objective(trial):
    start = time()  
    
    N_NEIGHBORS = trial.suggest_int('n_neighbors', low=1, high=9, step=2)
    
    # Using code from https://stackoverflow.com/a/58833843 to check for duplicate parameter set
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            raise optuna.exceptions.TrialPruned('Duplicate parameter set')

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
    if 'CAIX' in FLAGS.csv:
        dataset_label = 'DD1S_CAIX'
    elif 'sEH' in FLAGS.csv:
        dataset_label = 'triazine_sEH'
    elif 'SIRT2' in FLAGS.csv:
        dataset_label = 'triazine_SIRT2'

    # Extract counts
    exp_counts = np.array(df_data[[FLAGS.exp]], dtype='int')
    bead_counts = np.array(df_data[[FLAGS.beads]], dtype='int')
    exp_tot = np.sum(exp_counts, axis=0) # column sums
    bead_tot = np.sum(bead_counts, axis=0)
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
        train_slice, valid_slice, test_slice  = splitter(x, df_data, seed=FLAGS.seed)

    logging.info(f'Train length: {len(train_slice):12d}')
    logging.info(f'Valid length: {len(valid_slice):12d}')
    logging.info(f'Test length:  {len(test_slice):12d}')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: Train length: {len(train_slice):12d}\n')
        lf.write(f'{datetime.now()} INFO: Valid length: {len(valid_slice):12d}\n')
        lf.write(f'{datetime.now()} INFO: Test length:  {len(test_slice):12d}\n')
        
    true_labels_R = None

    # Load model
    saved_model = joblib.load(open(os.path.join(DELQSAR_ROOT, 'experiments', 'models', f'{dataset_label}', f'{model_label}', f'k_{N_NEIGHBORS}', f'{splitter_name}_seed_{FLAGS.seed}.joblib'), 'rb'))
    knn = models.kNN(task_type='regression', n_neighbors=N_NEIGHBORS, model=saved_model)
    
    if FLAGS.eval_metric == 'NLL':
        knn.eval_metric = losses.loss_fn_nlogprob
    elif FLAGS.eval_metric == 'MSE':
        knn.eval_metric = losses.loss_fn_MSE
    elif FLAGS.eval_metric == 'rank_corr_coeff':
        knn.eval_metric = metrics.get_spearman_r
    
    # Evaluations
    def evaluate_on_slice(test_slice, slice_label='Test'):
        if FLAGS.eval_metric == 'rank_corr_coeff':
            test_enrichments = knn.predict_on_x(
                x[test_slice, :], batch_size=1024,
            )
            k1 = torch.FloatTensor(exp_counts[test_slice])
            k2 = torch.FloatTensor(bead_counts[test_slice])
            n1 = float(exp_tot)
            n2 = float(bead_tot)
                
            corr = metrics.get_spearman_r(test_enrichments, k1, k2, n1, n2)
            
            formatted_corr = '{0:.5f}'.format(corr)
            logging.info(f'{slice_label} ({len(test_slice)} compounds) rank correlation coefficient = {formatted_corr}')
            with open(LOG_FILE, 'a') as lf:
                lf.write(f'{datetime.now()} INFO: {slice_label} ({len(test_slice)} compounds) rank correlation coefficient = {formatted_corr}\n')

            return corr

        # regression
        test_losses, test_enrichments = knn.evaluate_on_del(x, exp_counts, bead_counts, test_slice, true_labels=true_labels_R)

        avg_test_loss = np.sum(test_losses, axis=0) / test_losses.shape[0]    
        formatted_loss = ['{0:.5f}'.format(loss) for loss in avg_test_loss]

        logging.info(f'{slice_label} ({len(test_enrichments)} compounds) average loss = {np.squeeze(formatted_loss)}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: {slice_label} ({len(test_enrichments)} compounds) average loss = {np.squeeze(formatted_loss)}\n')

        R, R_lb, R_ub = R_ranges(bead_counts[test_slice], bead_tot, exp_counts[test_slice], exp_tot)

        low = np.mean(test_enrichments < R_lb)
        high = np.mean(test_enrichments > R_ub)
        within = np.mean((test_enrichments < R_ub) & (test_enrichments > R_lb))
        accuracy_vals = [low, high, within]

        # Report fraction of enrichments within (LB, UB)
        logging.info(f'{FLAGS.exp}')
        logging.info(f'% predicted R < calculated R: {accuracy_vals[0]:.3%}')
        logging.info(f'% predicted R > calculated R: {accuracy_vals[1]:.3%}')
        logging.info(f'% predicted R in (R_lb, R_ub): {accuracy_vals[2]:.3%}\n')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: {FLAGS.exp}\n')
            lf.write(f'{datetime.now()} INFO: % predicted R < calculated R: {accuracy_vals[0]:.3%}\n')
            lf.write(f'{datetime.now()} INFO: % predicted R > calculated R: {accuracy_vals[1]:.3%}\n')
            lf.write(f'{datetime.now()} INFO: % predicted R in (R_lb, R_ub): {accuracy_vals[2]:.3%}\n\n')

        return test_losses, test_enrichments, avg_test_loss, accuracy_vals
    
    # Evaluate on part of train slice
    try:
        evaluate_on_slice(train_slice[:5000], slice_label='Train subset')
    except ValueError as ve:
        logging.info(str(ve))
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} WARNING: {str(ve)}\n')
            
    # Evaluate on validation set
    if FLAGS.eval_metric == 'rank_corr_coeff':
        valid_corr = evaluate_on_slice(valid_slice, slice_label='Valid set')
    else:
        _valid_losses, _valid_enrichments, avg_valid_loss, valid_accuracy_vals = evaluate_on_slice(valid_slice, slice_label='Valid set')
            
    # Evaluate on test set
    if FLAGS.eval_metric == 'rank_corr_coeff':
        test_corr = evaluate_on_slice(test_slice, slice_label='Test set')
    else:
        _test_losses, _test_enrichments, avg_test_loss, test_accuracy_vals = evaluate_on_slice(test_slice, slice_label='Test set')

    # Record to results file
    if not os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as fid:
            cols = ['Time (of recording)']
            for f in FLAGS.get_key_flags_for_module(sys.argv[0]):
                flag,value = f.serialize().split('=')
                cols.append(flag[2:])
            cols.append('Training losses')
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
        if FLAGS.eval_metric in ['NLL', 'MSE']:
            cols.append(str(np.squeeze(avg_valid_loss)))
            cols.append('')
            cols.append('{0:.3%}'.format(valid_accuracy_vals[2]))
            cols.append('')
            cols.append('')
            cols.append(str(np.squeeze(avg_test_loss)))
            cols.append('')
            cols.append('{0:.3%}'.format(test_accuracy_vals[2]))
            cols.append('')
            cols.append('')
        else: # rank corr coeff
            cols.append('')
            cols.append(str(valid_corr))
            cols.append('')
            cols.append('')
            cols.append('')
            cols.append('')
            cols.append(str(test_corr))
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
        return valid_corr
    else:
        return avg_valid_loss
  
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
