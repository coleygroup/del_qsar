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

from del_qsar import models, featurizers, splitters, metrics
from del_qsar.enrichments import R_from_z, R_ranges


FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'DD1S_CAIX_QSAR.csv', 'csv filename')
flags.DEFINE_string('hyperparams', None, 'csv with hyperparameters of saved models')
flags.DEFINE_string('fps_h5', 'x_DD1S_CAIX_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_list('exp', ['exp_tot'], 'Column header(s) for data counts: experiment')
flags.DEFINE_list('beads', ['beads_tot'], 'Column header(s) for data counts: beads')
flags.DEFINE_string('model_type', 'FP-FFNN', 'Model type')
flags.DEFINE_string('out', None, 'Experiment label (subfolder)')
flags.DEFINE_string('device', 'cuda:0', 'Device (set to cuda:0 if using GPU)')
flags.DEFINE_integer('num_thresholds', 20, 'Number of thresholds')
flags.DEFINE_integer('start_idx', 0, 'Start index (if resuming a job)')
flags.DEFINE_integer('stop_idx', 20, 'Stop index')
flags.DEFINE_boolean('random_split_only', True, 'Evaluate for random split only')
flags.DEFINE_boolean('random_guess', False, 'Random-guess baseline')
flags.DEFINE_integer('num_workers', 20, 'Number of workers for loading data (for D-MPNNs)')

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
    
    if 'DD1S_CAIX' in FLAGS.csv and not FLAGS.random_guess:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'DD1S_CAIX_{FLAGS.model_type}_bin_eval_{FLAGS.num_thresholds}_thresholds.csv')
    elif 'DD1S_CAIX' in FLAGS.csv and FLAGS.random_guess:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'DD1S_CAIX_random_guess_bin_eval_{FLAGS.num_thresholds}_thresholds.csv')
        
    elif 'sEH' in FLAGS.exp[0] and not FLAGS.random_guess:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_sEH_{FLAGS.model_type}_bin_eval_{FLAGS.num_thresholds}_thresholds.csv')
    elif 'sEH' in FLAGS.exp[0] and FLAGS.random_guess:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_sEH_random_guess_bin_eval_{FLAGS.num_thresholds}_thresholds.csv')
        
    elif 'SIRT2' in FLAGS.exp[0] and not FLAGS.random_guess:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_SIRT2_{FLAGS.model_type}_bin_eval_{FLAGS.num_thresholds}_thresholds.csv')
    elif 'SIRT2' in FLAGS.exp[0] and FLAGS.random_guess:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_SIRT2_random_guess_bin_eval_{FLAGS.num_thresholds}_thresholds.csv')
        
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
    if not ('D-MPNN' in FLAGS.model_type and 'triazine' in FLAGS.csv):
        hyperparams = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments', FLAGS.hyperparams))
        num_replicates = 5
    else:
        num_replicates = 3
    if 'triazine' in FLAGS.csv:
        for col in df_data.columns:
            if (' ' in col):
                df_data = df_data.rename(columns={col: col.replace(' ', '_')})
    
    # Extract counts and get calculated enrichments
    exp_counts = np.array(df_data[FLAGS.exp], dtype='int')
    bead_counts = np.array(df_data[FLAGS.beads], dtype='int')
    exp_tot = np.sum(exp_counts, axis=0) # column sums
    bead_tot = np.sum(bead_counts, axis=0)
    R, R_lb, R_ub = R_ranges(bead_counts[:, 0], bead_tot[0], exp_counts[:, 0], exp_tot[0])
    
    # Get featurization
    if 'D-MPNN' in FLAGS.model_type:
        smis = df_data['smiles']
        targets = []
        for i in range(exp_counts.shape[0]):
            targets_for_compound = []
            for j in range(exp_counts.shape[1]): 
                targets_for_compound.append(R_from_z(bead_counts[i, j], bead_tot[j], 
                                                     exp_counts[i, j], exp_tot[j], 0).tolist())
            targets.append(targets_for_compound)
        featurizer = featurizers.GraphFeaturizer(smis, targets)
        x = featurizer.prepare_x()
    elif 'FP-FFNN' in FLAGS.model_type:
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
    elif 'OH-FFNN' in FLAGS.model_type:
        featurizer = featurizers.OneHotFeaturizer(df_data)
        x = featurizer.prepare_x(df_data)
          
    # Get predicted enrichments for each data split and replicate
    splitter_names = [
        'random',
        'cycle1',
        'cycle2',
        'cycle3',
        'cycle12',
        'cycle13',
        'cycle23',
        'cycle123',
    ]
    if FLAGS.random_split_only:
        _splitters = [splitters.RandomSplitter()]
    else:
        _splitters = [
            splitters.RandomSplitter(),
            splitters.OneCycleSplitter(['cycle1'], LOG_FILE),
            splitters.OneCycleSplitter(['cycle2'], LOG_FILE),
            splitters.OneCycleSplitter(['cycle3'], LOG_FILE),
            splitters.TwoCycleSplitter(['cycle1','cycle2'], LOG_FILE),
            splitters.TwoCycleSplitter(['cycle1','cycle3'], LOG_FILE),
            splitters.TwoCycleSplitter(['cycle2','cycle3'], LOG_FILE),
            splitters.ThreeCycleSplitter(['cycle1','cycle2','cycle3'], LOG_FILE),
        ]
        
    # Get thresholds
    if 'DD1S_CAIX' in FLAGS.csv:
        min_top_percent = (2000/21706) # threshold at which the expected number of positives in the test set is 20
        max_top_percent = 10
    elif 'triazine' in FLAGS.csv:
        min_top_percent = (2000/1131000) # threshold at which the expected number of positives in the test set is 20
        max_top_percent = 2
    
    top_percents = np.geomspace(min_top_percent, max_top_percent, num=FLAGS.num_thresholds)
    
    for i in range(len(_splitters)):
        logging.info(f'{splitter_names[i]} split')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'\n{datetime.now()} INFO: {splitter_names[i]} split\n\n')
        
        for top_percent in top_percents[FLAGS.start_idx:FLAGS.stop_idx]:
            pr_aucs_R = []
            roc_aucs_R = []
            
            for seed in range(num_replicates):
                torch.manual_seed(seed)
                logging.info(f'Seed {seed}')
                with open(LOG_FILE, 'a') as lf:
                    lf.write(f'{datetime.now()} INFO: Seed {seed}\n')

                train_slice, valid_slice, test_slice = _splitters[i](x, df_data, seed=seed)

                # Get enrichments
                if 'D-MPNN' in FLAGS.model_type and 'triazine' in FLAGS.csv: # triazine D-MPNN
                    model = models.MoleculeModel(depth=6, hidden_size=1500, ffn_num_layers=3, 
                                         dropout=0.05, device=FLAGS.device, torch_seed=seed)
                    if 'sEH' in FLAGS.exp[0]:
                        model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                                      f'models/triazine_sEH/{FLAGS.model_type}', 
                                                                      f'{splitter_names[i]}_seed_{seed}.torch')))
                    elif 'SIRT2' in FLAGS.exp[0]:
                        model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                                      f'models/triazine_SIRT2/{FLAGS.model_type}', 
                                                                      f'{splitter_names[i]}_seed_{seed}.torch')))
                    model = model.to(FLAGS.device)

                    test_enrichments = model.predict_on_x(
                        [x[idx] for idx in test_slice], batch_size=50,
                        device=FLAGS.device,
                        num_workers=FLAGS.num_workers,
                    )

                elif 'D-MPNN' in FLAGS.model_type: # DD1S CAIX D-MPNN
                    df_row = hyperparams[
                        hyperparams['dataset'].isin(['DD1S_CAIX']) &
                        hyperparams['model type'].isin([str(FLAGS.model_type)]) &
                        hyperparams['seed'].isin([seed]) &
                        hyperparams['split'].isin([str(splitter_names[i])]) 
                    ]
                    depth = int(df_row['depth'])
                    hidden_size = int(df_row['hidden size'])
                    ffn_num_layers = int(df_row['FFN num layers'])
                    dropout = float(df_row['dropout'])

                    model = models.MoleculeModel(depth=depth, hidden_size=hidden_size, ffn_num_layers=ffn_num_layers, 
                                                 dropout=dropout, device=FLAGS.device, torch_seed=seed)
                    model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                                  f'models/DD1S_CAIX/{FLAGS.model_type}', 
                                                                  f'{splitter_names[i]}_seed_{seed}.torch')))
                    model = model.to(FLAGS.device)

                    test_enrichments = model.predict_on_x(
                        [x[idx] for idx in test_slice], batch_size=50,
                        device=FLAGS.device,
                        num_workers=FLAGS.num_workers,
                    )

                elif 'DD1S_CAIX' in FLAGS.csv: # DD1S CAIX FFNN
                    df_row = hyperparams[
                        hyperparams['dataset'].isin(['DD1S_CAIX']) &
                        hyperparams['model type'].isin([str(FLAGS.model_type)]) &
                        hyperparams['seed'].isin([seed]) &
                        hyperparams['split'].isin([str(splitter_names[i])]) 
                    ]

                    layer_sizes = [int(s) for s in list(df_row['layer sizes'])[0].split(', ')]
                    dropout = float(df_row['dropout'])
                    input_size = x.shape[1]

                    model = models.MLP(input_size, [int(size) for size in layer_sizes], dropout=dropout, torch_seed=seed)
                    model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                                  f'models/DD1S_CAIX/{FLAGS.model_type}', 
                                                                  f'{splitter_names[i]}_seed_{seed}.torch')))
                    model = model.to(FLAGS.device)

                    test_enrichments = model.predict_on_x(
                        x[test_slice, :], batch_size=1024,
                        device=FLAGS.device,
                    )

                elif 'sEH' in FLAGS.exp[0]: # triazine sEH FFNN
                    df_row = hyperparams[
                        hyperparams['dataset'].isin(['triazine_sEH']) &
                        hyperparams['model type'].isin([str(FLAGS.model_type)]) &
                        hyperparams['seed'].isin([seed]) &
                        hyperparams['split'].isin([str(splitter_names[i])]) 
                    ]
                    layer_sizes = [int(s) for s in list(df_row['layer sizes'])[0].split(', ')]
                    dropout = float(df_row['dropout'])
                    input_size = x.shape[1]

                    model = models.MLP(input_size, [int(size) for size in layer_sizes], dropout=dropout, torch_seed=seed)
                    model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                                  f'models/triazine_sEH/{FLAGS.model_type}', 
                                                                  f'{splitter_names[i]}_seed_{seed}.torch')))
                    model = model.to(FLAGS.device)

                    test_enrichments = model.predict_on_x(
                        x[test_slice, :], batch_size=1024,
                        device=FLAGS.device,
                    )

                elif 'SIRT2' in FLAGS.exp[0]: # triazine SIRT2 FFNN
                    df_row = hyperparams[
                        hyperparams['dataset'].isin(['triazine_SIRT2']) &
                        hyperparams['model type'].isin([str(FLAGS.model_type)]) &
                        hyperparams['seed'].isin([seed]) &
                        hyperparams['split'].isin([str(splitter_names[i])]) 
                    ]
                    layer_sizes = [int(s) for s in list(df_row['layer sizes'])[0].split(', ')]
                    dropout = float(df_row['dropout'])
                    input_size = x.shape[1]

                    model = models.MLP(input_size, [int(size) for size in layer_sizes], dropout=dropout, torch_seed=seed)
                    model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                                  f'models/triazine_SIRT2/{FLAGS.model_type}', 
                                                                  f'{splitter_names[i]}_seed_{seed}.torch')))
                    model = model.to(FLAGS.device)

                    test_enrichments = model.predict_on_x(
                        x[test_slice, :], batch_size=1024,
                        device=FLAGS.device,
                    )

                logging.info(str(model))
                with open(LOG_FILE, 'a') as lf:
                    lf.write(f'{datetime.now()} INFO: ' + str(model) + '\n')

                # Get threshold
                threshold_R = np.percentile(R[train_slice], 100-top_percent)
                logging.info(f'Top percent: {top_percent}')
                logging.info(f'Threshold (R): {threshold_R}')
                with open(LOG_FILE, 'a') as lf:
                    lf.write(f'{datetime.now()} INFO: Top percent: {top_percent}\n')
                    lf.write(f'{datetime.now()} INFO: Threshold (R): {threshold_R}\n')

                # Get ground truth labels for test set
                true_labels_R = np.zeros((len(test_slice),), dtype='int')
                for idx, ti in enumerate(test_slice):
                    if R[ti] > threshold_R:
                        true_labels_R[idx] = 1
                                   
                # For random guess: shuffle predictions                       
                if FLAGS.random_guess:
                    np.random.seed(seed)
                    np.random.shuffle(test_enrichments)

                # Get PR AUC and ROC AUC scores
                pr_auc_R = metrics.get_pr_auc(true_labels_R, test_enrichments)
                pr_aucs_R.append(pr_auc_R) 
                
                roc_auc_R = metrics.get_roc_auc(true_labels_R, test_enrichments)
                roc_aucs_R.append(roc_auc_R)

                logging.info(f'PR AUC (R):  {"{0:.5f}".format(pr_auc_R)}')  
                logging.info(f'ROC AUC (R): {"{0:.5f}".format(roc_auc_R)}') 
                with open(LOG_FILE, 'a') as lf:      
                    lf.write(f'{datetime.now()} INFO: PR AUC (R):  {"{0:.5f}".format(pr_auc_R)}\n')
                    lf.write(f'{datetime.now()} INFO: ROC AUC (R): {"{0:.5f}".format(roc_auc_R)}\n')

                # Record
                if not os.path.isfile(RESULTS_FILE):
                    with open(RESULTS_FILE, 'w') as fid:
                        cols = ['Time (of recording)']
                        cols.append('Data split')
                        cols.append('Top percent')
                        cols.append('Random seed')
                        cols.append('PR AUC')
                        cols.append('ROC AUC')
                        fid.write('\t'.join(cols) + '\n')
                with open(RESULTS_FILE, 'a') as fid:
                    cols = [str(datetime.now().now())]
                    cols.append(str(splitter_names[i]))
                    cols.append(str(top_percent))
                    cols.append(str(seed))
                    cols.append(str(pr_auc_R))
                    cols.append(str(roc_auc_R))
                    fid.write('\t'.join(cols) + '\n')
                fid.close()

            avg_pr_auc_R = sum(pr_aucs_R) / len(pr_aucs_R)
            avg_roc_auc_R = sum(roc_aucs_R) / len(roc_aucs_R)

            logging.info(f'Average PR AUC (R):  {"{0:.5f}".format(avg_pr_auc_R)}')                      
            logging.info(f'Average ROC AUC (R): {"{0:.5f}".format(avg_roc_auc_R)}\n')    
            with open(LOG_FILE, 'a') as lf: 
                lf.write(f'{datetime.now()} INFO: Average PR AUC (R):  {"{0:.5f}".format(avg_pr_auc_R)}\n')
                lf.write(f'{datetime.now()} INFO: Average ROC AUC (R): {"{0:.5f}".format(avg_roc_auc_R)}\n\n')
            
    lf.close()
    
if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
 