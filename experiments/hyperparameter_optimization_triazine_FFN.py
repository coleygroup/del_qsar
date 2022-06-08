from timeit import default_timer as time
import os
import sys
import logging
from datetime import datetime
import optuna
import pandas as pd
import numpy as np
import torch
import h5py

from absl import app
from absl import flags

import rdkit
print(rdkit.__version__)

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

RESULTS_FILE = os.path.join(DELQSAR_ROOT, 'experiments', 'all_results.csv')

from del_qsar import featurizers, splitters, models, losses
from del_qsar.enrichments import R_ranges


FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'triazine_lib_sEH_SIRT2_QSAR.csv', 'csv filename')
flags.DEFINE_string('fps_h5', 'x_triazine_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_list('exp', ['sEH_[strep]_tot'], 'Column header(s) for data counts: experiment')
flags.DEFINE_list('beads', ['beads-linker-only_[strep]_tot'], 'Column header(s) for data counts: beads')
flags.DEFINE_enum('featurizer', 'fingerprint', ['fingerprint', 'onehot'], 'How molecules are featurized')
flags.DEFINE_list('splitter', ['random'], 'How molecules are split for training/testing') 
flags.DEFINE_integer('seed', 0, 'Random seed for data splitting and weight initialization')

flags.DEFINE_enum('model_type', 'MLP', ['MLP'], 'Model type')
flags.DEFINE_enum('task_type', 'regression', ['regression', 'classification'], 'Task type')

flags.DEFINE_enum('threshold_type', 'percentile', ['percentile', 'fixed'], 'Threshold type (for classification only)')
flags.DEFINE_float('threshold_val', 99.99, 'Threshold value; exact value or percentile (for classification only)')

flags.DEFINE_enum('loss_fn_train', 'nlogprob', ['nlogprob', 'MSE', 'BCE'], 
                  'Loss function during training (note: classifiers automatically use BCE)')

flags.DEFINE_integer('max_epochs', 300, 'Maximum number of epochs')
flags.DEFINE_integer('patience', 5, 'Patience')
flags.DEFINE_float('max_norm', 5, 'Max norm')

flags.DEFINE_string('out', None, 'Experiment label (subfolder)')

flags.DEFINE_string('device', 'cuda:0', 'Device (set to cuda:0 if using GPU)')

flags.DEFINE_integer('n_trials', 25, 'Number of optuna trials to run')

if not os.path.isdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results')):
    os.mkdir(os.path.join(DELQSAR_ROOT, 'experiments', 'results'))
    
dt = datetime.today()
DATE = os.path.join(DELQSAR_ROOT, 'experiments', 'results', 
                    f'{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}')
if not os.path.isdir(DATE):
    os.mkdir(DATE)

    
def objective(trial):
    start = time()
    
    INIT_LR = trial.suggest_float('init_lr', low=1e-5, high=3e-2, log=True)
    
    NUM_LAYERS = trial.suggest_int('num_layers', low=1, high=3, step=1)
    MIN_LAYER_SIZE_EXP = trial.suggest_int('min_layer_size_exponent', low=6, high=10, step=1) # smallest layer size: 64 to 1024
    if NUM_LAYERS > 1:
        DROPOUT = trial.suggest_discrete_uniform('dropout', low=0.0, high=0.5, q=0.05)
        
        if NUM_LAYERS == 2 and MIN_LAYER_SIZE_EXP <= 8: # two layers with min layer size <= 256
            TWO_LAYERS_TYPE_3 = trial.suggest_categorical('two_layers_architecture_3', ['flat', 'pyramid_2', 'pyramid_4'])
            if TWO_LAYERS_TYPE_3 == 'flat':
                LAYER_SIZES = [2**MIN_LAYER_SIZE_EXP, 2**MIN_LAYER_SIZE_EXP]
            elif TWO_LAYERS_TYPE_3 == 'pyramid_2':
                LAYER_SIZES = [2**(MIN_LAYER_SIZE_EXP + 1), 2**MIN_LAYER_SIZE_EXP]
            elif TWO_LAYERS_TYPE_3 == 'pyramid_4':
                LAYER_SIZES = [2**(MIN_LAYER_SIZE_EXP + 2), 2**MIN_LAYER_SIZE_EXP]
            
        elif NUM_LAYERS == 2 and MIN_LAYER_SIZE_EXP == 9: # two layers with min layer size 512
            TWO_LAYERS_TYPE_2 = trial.suggest_categorical('two_layers_architecture_2', ['flat', 'pyramid_2'])
            if TWO_LAYERS_TYPE_2 == 'flat':
                LAYER_SIZES = [512, 512]
            elif TWO_LAYERS_TYPE_2 == 'pyramid_2':
                LAYER_SIZES = [1024, 512]
        
        elif NUM_LAYERS == 2 and MIN_LAYER_SIZE_EXP == 10: # two layers with min layer size 1024
            LAYER_SIZES = [1024, 1024]
        
        elif NUM_LAYERS == 3 and MIN_LAYER_SIZE_EXP <= 6: # three layers with min layer size <= 64
            THREE_LAYERS_TYPE_3 = trial.suggest_categorical('three_layers_architecture_3', ['flat', 'pyramid_2', 'pyramid_4'])
            if THREE_LAYERS_TYPE_3 == 'flat':
                LAYER_SIZES = [2**MIN_LAYER_SIZE_EXP, 2**MIN_LAYER_SIZE_EXP, 2**MIN_LAYER_SIZE_EXP]
            elif THREE_LAYERS_TYPE_3 == 'pyramid_2':
                LAYER_SIZES = [2**(MIN_LAYER_SIZE_EXP + 2), 2**(MIN_LAYER_SIZE_EXP + 1), 2**MIN_LAYER_SIZE_EXP]
            elif THREE_LAYERS_TYPE_3 == 'pyramid_4':
                LAYER_SIZES = [2**(MIN_LAYER_SIZE_EXP + 4), 2**(MIN_LAYER_SIZE_EXP + 2), 2**MIN_LAYER_SIZE_EXP]
        
        elif NUM_LAYERS == 3 and MIN_LAYER_SIZE_EXP in [7, 8]: # three layers with min layer size 128 or 256
            THREE_LAYERS_TYPE_2 = trial.suggest_categorical('three_layers_architecture_2', ['flat', 'pyramid_2'])
            if THREE_LAYERS_TYPE_2 == 'flat':
                LAYER_SIZES = [2**MIN_LAYER_SIZE_EXP, 2**MIN_LAYER_SIZE_EXP, 2**MIN_LAYER_SIZE_EXP]
            elif THREE_LAYERS_TYPE_2 == 'pyramid_2':
                LAYER_SIZES = [2**(MIN_LAYER_SIZE_EXP + 2), 2**(MIN_LAYER_SIZE_EXP + 1), 2**MIN_LAYER_SIZE_EXP]
        
        elif NUM_LAYERS == 3 and MIN_LAYER_SIZE_EXP > 8: # three layers with min layer size 512 or 1024
            LAYER_SIZES = [2**MIN_LAYER_SIZE_EXP, 2**MIN_LAYER_SIZE_EXP, 2**MIN_LAYER_SIZE_EXP]
        
    else: # one layer
        DROPOUT = None
        LAYER_SIZES = [2**MIN_LAYER_SIZE_EXP]                              

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
        # GPU?
        logging.info(f'CUDA available? {torch.cuda.is_available()}')
        lf.write(f'{datetime.now()} INFO: CUDA available? {torch.cuda.is_available()}\n')

    # Get data
    df_data = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments', 'datasets', FLAGS.csv))
    for col in df_data.columns:
        if ' ' in col:
            df_data = df_data.rename(columns={col: col.replace(' ', '_')})

    # Extract counts
    exp_counts = np.array(df_data[FLAGS.exp], dtype='int')
    bead_counts = np.array(df_data[FLAGS.beads], dtype='int')
    exp_tot = np.sum(exp_counts, axis=0) # column sums
    bead_tot = np.sum(bead_counts, axis=0)
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
    else:
        raise ValueError('Unknown featurizer')
        
    INPUT_SIZE = x.shape[1] 
    logging.info(f'Input featurization is {INPUT_SIZE} long')
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: Input featurization is {INPUT_SIZE} long\n')

    # Define different splits
    if FLAGS.splitter[0] == 'random':
        splitter = splitters.RandomSplitter()
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
        elif len(FLAGS.splitter) == 2:
            splitter = splitters.TwoCycleSplitter(FLAGS.splitter, LOG_FILE)
        elif len(FLAGS.splitter) == 3:
            splitter = splitters.ThreeCycleSplitter(FLAGS.splitter, LOG_FILE)
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

    # For binary classification: get ground truth labels
    if FLAGS.task_type == 'classification':
        R, R_lb, R_ub = R_ranges(bead_counts[:, 0], bead_tot[0], exp_counts[:, 0], exp_tot[0])
        if FLAGS.threshold_type == 'percentile':
            threshold_R = np.percentile(R[train_slice], FLAGS.threshold_val)
        else:
            threshold_R = FLAGS.threshold_val
        logging.info(f'Threshold (R): {threshold_R}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: Threshold (R): {threshold_R}\n')
        true_labels_R = torch.zeros((len(R),))
        for idx in range(len(R)):
            if R[idx] > threshold_R:
                true_labels_R[idx] = 1
    else:
        true_labels_R = None

        
    # Build model and train
    BATCH_SIZE = 1024
    NUM_TASKS = len(FLAGS.exp)
    
    torch.manual_seed(FLAGS.seed)  
    if FLAGS.task_type == 'classification':
        model = models.MLP(INPUT_SIZE, [int(size) for size in LAYER_SIZES],
                dropout=DROPOUT, num_tasks=NUM_TASKS, torch_seed=FLAGS.seed, 
                task_type='classification')
    else:
        model = models.MLP(INPUT_SIZE, [int(size) for size in LAYER_SIZES],
                dropout=DROPOUT, num_tasks=NUM_TASKS, torch_seed=FLAGS.seed)
     
    model.optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
                        
    # Loss function during training
    if FLAGS.task_type == 'classification':
        model.loss_fn_train = losses.loss_fn_BCE
    elif FLAGS.loss_fn_train == 'nlogprob':
        model.loss_fn_train = losses.loss_fn_nlogprob
    elif FLAGS.loss_fn_train == 'MSE': 
        model.loss_fn_train = losses.loss_fn_MSE
    else:
        raise ValueError('Unknown loss function for training')    
    
    # Loss function during evaluation (only used for regression models)
    model.eval_metric = losses.loss_fn_nlogprob
    
    logging.info(str(model))
    with open(LOG_FILE, 'a') as lf:
        lf.write(f'{datetime.now()} INFO: ' + str(model) + '\n')
        # Move to GPU?
        model.to(torch.device(FLAGS.device))
        logging.info(f'Moved model to {FLAGS.device}')
        lf.write(f'{datetime.now()} INFO: Moved model to {FLAGS.device}\n')

    try:
        model.train_on_del(
            x, exp_counts, bead_counts, train_slice, valid_slice, true_labels=true_labels_R,
            batch_size=BATCH_SIZE, max_epochs=FLAGS.max_epochs, 
            patience=FLAGS.patience, max_norm=FLAGS.max_norm,
            zscale=lambda epoch: 1 + 9*np.exp(-epoch/2),
            device=FLAGS.device, output_size=NUM_TASKS,
            save_path=os.path.join(SAVE_ROOT, f'best_model_trial_{trial.number}.torch'),
            log_path=LOG_FILE,
            torch_seed=FLAGS.seed,
        )
    except KeyboardInterrupt:
        logging.warning('Training interrupted by KeyboardInterrupt!')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} WARNING: Training interrupted by KeyboardInterrupt!\n')
            
    # Check for NaN loss
    if model.best_val_loss == float('inf'):
        return model.best_val_loss
    
    # Record loss functions
    with open(os.path.join(SAVE_ROOT, f'losses_trial_{trial.number}.csv'), 'w') as fid:
        fid.write('epoch,train_loss,valid_loss\n')
        for i in range(len(model.all_valid_losses)):
            fid.write(f'{i+1},{np.squeeze(model.all_train_losses[i])}, {np.squeeze(model.all_valid_losses[i])}\n')
    train_losses_row_sums = np.sum(model.all_train_losses, axis=1)
    valid_losses_row_sums = np.sum(model.all_valid_losses, axis=1)
  

    # Evaluations
    def evaluate_on_slice(test_slice, slice_label='Test'):
        
        if model.classification:
            test_roc_auc, test_pr_auc, test_preds = model.evaluate_on_del(
                x, exp_counts, bead_counts, test_slice, batch_size=BATCH_SIZE,
                device=FLAGS.device, true_labels=true_labels_R,
            )
            
            formatted_roc_auc = ['{0:.5f}'.format(roc_auc) for roc_auc in np.squeeze(test_roc_auc, axis=0)]
            formatted_pr_auc = ['{0:.5f}'.format(pr_auc) for pr_auc in np.squeeze(test_pr_auc, axis=0)]
            logging.info(f'{slice_label} ({len(test_preds)} compounds) ROC AUC = {np.squeeze(formatted_roc_auc)}')
            logging.info(f'{slice_label} ({len(test_preds)} compounds) PR AUC = {np.squeeze(formatted_pr_auc)}')
            with open(LOG_FILE, 'a') as lf:
                lf.write(f'{datetime.now()} INFO: {slice_label} ({len(test_preds)} compounds) ROC AUC = {np.squeeze(formatted_roc_auc)}\n')
                lf.write(f'{datetime.now()} INFO: {slice_label} ({len(test_preds)} compounds) PR AUC = {np.squeeze(formatted_pr_auc)}\n')
            return test_roc_auc, test_pr_auc, test_preds

        test_losses, test_enrichments = model.evaluate_on_del(
            x, exp_counts, bead_counts, test_slice, batch_size=BATCH_SIZE,
            device=FLAGS.device,
        )

        avg_test_loss = np.sum(test_losses, axis=0) / test_losses.shape[0]
        formatted_loss = ['{0:.5f}'.format(loss) for loss in avg_test_loss]
        logging.info(f'{slice_label} ({len(test_enrichments)} compounds) average loss = {np.squeeze(formatted_loss)}')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: {slice_label} ({len(test_enrichments)} compounds) average loss = {np.squeeze(formatted_loss)}\n')

        R = np.zeros((NUM_TASKS, len(test_slice)))
        R_lb = np.zeros((NUM_TASKS, len(test_slice)))
        R_ub = np.zeros((NUM_TASKS, len(test_slice)))
        for i in range(NUM_TASKS):
            _R, _R_lb, _R_ub = R_ranges(bead_counts[test_slice, i], bead_tot[i], exp_counts[test_slice, i], exp_tot[i])
            R[i] = _R
            R_lb[i] = _R_lb
            R_ub[i] = _R_ub

        test_loss_by_POI = []
        test_enrichment_by_POI = []
        for col in test_losses.T:
            test_loss_by_POI.append(col.tolist())
        for col in test_enrichments.T:
            test_enrichment_by_POI.append(col.tolist())
        test_enrichment_by_POI = np.array(test_enrichment_by_POI)

        accuracy_vals = np.zeros((NUM_TASKS, 3))
        for i in range(NUM_TASKS):
            low = np.mean(test_enrichment_by_POI[i] < R_lb[i])
            high = np.mean(test_enrichment_by_POI[i] > R_ub[i])
            within = np.mean((test_enrichment_by_POI[i] < R_ub[i]) & (test_enrichment_by_POI[i] > R_lb[i]))
            accuracy_vals[i] = [low, high, within]

        # Report fraction of enrichments within (LB, UB)
        for i, POI in enumerate(FLAGS.exp):
            logging.info(f'{POI}')
            logging.info(f'% predicted R < calculated R: {accuracy_vals[i, 0]:.3%}')
            logging.info(f'% predicted R > calculated R: {accuracy_vals[i, 1]:.3%}')
            logging.info(f'% predicted R in (R_lb, R_ub): {accuracy_vals[i, 2]:.3%}')
        with open(LOG_FILE, 'a') as lf:
            for i, POI in enumerate(FLAGS.exp):
                lf.write(f'{datetime.now()} INFO: {POI}\n')
                lf.write(f'{datetime.now()} INFO: % predicted R < calculated R: {accuracy_vals[i, 0]:.3%}\n')
                lf.write(f'{datetime.now()} INFO: % predicted R > calculated R: {accuracy_vals[i, 1]:.3%}\n')
                lf.write(f'{datetime.now()} INFO: % predicted R in (R_lb, R_ub): {accuracy_vals[i, 2]:.3%}\n\n')

        return test_losses, test_enrichments, avg_test_loss, accuracy_vals
        
    # Evaluate on part of train slice
    try:
        evaluate_on_slice(train_slice[:5000], slice_label='Train subset')
    except ValueError as ve:
        logging.info(str(ve))
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'{datetime.now()} WARNING: {str(ve)}\n')
    
    # Evaluate on test set
    if model.classification:
        test_roc_auc, test_pr_auc, test_preds = evaluate_on_slice(test_slice, slice_label='Test set')
    else:
        _losses, _enrichments, avg_test_loss, accuracy_vals = evaluate_on_slice(test_slice, slice_label='Test set')

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
        cols.append(str(list(np.squeeze(model.all_train_losses))))
        cols.append(str(list(np.squeeze(model.all_valid_losses))))
        cols.append('')
        cols.append('')
        cols.append('')
        cols.append('')
        if model.classification:
            cols.append('')
            cols.append('')
            cols.append('')
            cols.append(str(np.squeeze(test_roc_auc)))
            cols.append(str(np.squeeze(test_pr_auc)))
        else:
            cols.append(str(np.squeeze(avg_test_loss)))
            cols.append('')
            acc_vals = accuracy_vals[:, 2]
            formatted_acc_vals = ['{0:.3%}'.format(val) for val in acc_vals]
            cols.append(f'{np.squeeze(formatted_acc_vals)}')
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
    
    return model.best_val_loss 
  
def main(argv):
    del argv
    study = optuna.create_study()
    study.optimize(objective, n_trials=FLAGS.n_trials)
    
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
    
    os.rename(os.path.join(SAVE_ROOT, f'losses_trial_{trial.number}.csv'), 
              os.path.join(SAVE_SUBFOLDER, f'losses_trial_{trial.number}.csv'))
    os.rename(os.path.join(SAVE_ROOT, f'best_model_trial_{trial.number}.torch'), 
              os.path.join(SAVE_SUBFOLDER, f'best_model_trial_{trial.number}.torch'))
    
if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
