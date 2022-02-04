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

from del_qsar import models, featurizers, splitters, losses
from del_qsar.enrichments import R_from_z


FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'DD1S_CAIX_QSAR.csv', 'csv filename')
flags.DEFINE_string('hyperparams', None, 'csv with hyperparameters of saved models')
flags.DEFINE_string('fps_h5', 'x_DD1S_CAIX_2048_bits_all_fps.h5', 'HDF5 file with stored fingerprints')
flags.DEFINE_list('exp', ['exp_tot'], 'Column header(s) for data counts: experiment')
flags.DEFINE_list('beads', ['beads_tot'], 'Column header(s) for data counts: beads')
flags.DEFINE_string('model_type', 'FP-FFNN', 'Model type')
flags.DEFINE_string('out', None, 'Experiment label (subfolder)')
flags.DEFINE_string('device', 'cuda:0', 'Device (set to cuda:0 if using GPU)')
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
    
    if 'DD1S_CAIX' in FLAGS.csv:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'DD1S_CAIX_{FLAGS.model_type}_MSE_loss_eval.csv')
    elif 'sEH' in FLAGS.exp[0]:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_sEH_{FLAGS.model_type}_MSE_loss_eval.csv')
    elif 'SIRT2' in FLAGS.exp[0]:
        RESULTS_FILE = os.path.join(SAVE_ROOT, f'triazine_SIRT2_{FLAGS.model_type}_MSE_loss_eval.csv')
        
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
            if ' ' in col:
                df_data = df_data.rename(columns={col: col.replace(' ', '_')})
    
    # Extract counts and get calculated enrichments
    exp_counts = np.array(df_data[FLAGS.exp], dtype='int')
    bead_counts = np.array(df_data[FLAGS.beads], dtype='int')
    exp_tot = np.sum(exp_counts, axis=0) # column sums
    bead_tot = np.sum(bead_counts, axis=0)
    
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

    all_splitters = [
        splitters.RandomSplitter(),
        splitters.OneCycleSplitter(['cycle1'], LOG_FILE),
        splitters.OneCycleSplitter(['cycle2'], LOG_FILE),
        splitters.OneCycleSplitter(['cycle3'], LOG_FILE),
        splitters.TwoCycleSplitter(['cycle1','cycle2'], LOG_FILE),
        splitters.TwoCycleSplitter(['cycle1','cycle3'], LOG_FILE),
        splitters.TwoCycleSplitter(['cycle2','cycle3'], LOG_FILE),
        splitters.ThreeCycleSplitter(['cycle1','cycle2','cycle3'], LOG_FILE),
    ]
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
    
    for i in range(len(all_splitters)):
        logging.info(f'{splitter_names[i]} split')
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'\n{datetime.now()} INFO: {splitter_names[i]} split\n\n')
            
        mse_losses = []
        
        for seed in range(num_replicates):
            torch.manual_seed(seed)
            logging.info(f'Seed {seed}')
            with open(LOG_FILE, 'a') as lf:
                lf.write(f'{datetime.now()} INFO: Seed {seed}\n')
                
            train_slice, valid_slice, test_slice = all_splitters[i](x, df_data, seed=seed)

            # Get enrichments
            if 'D-MPNN' in FLAGS.model_type and 'triazine' in FLAGS.csv: # triazine D-MPNN
                model = models.MoleculeModel(depth=6, hidden_size=1500, ffn_num_layers=3, 
                                     dropout=0.05, num_tasks=len(FLAGS.exp), device=FLAGS.device, torch_seed=seed)
                if 'sEH' in FLAGS.exp[0] and len(FLAGS.exp) == 1:
                    model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                                  f'models/triazine_sEH/{FLAGS.model_type}', 
                                                                  f'{splitter_names[i]}_seed_{seed}.torch')))
                elif 'SIRT2' in FLAGS.exp[0] and len(FLAGS.exp) == 1:
                    model.load_state_dict(torch.load(os.path.join(DELQSAR_ROOT, 'experiments',
                                                                  f'models/triazine_SIRT2/{FLAGS.model_type}', 
                                                                  f'{splitter_names[i]}_seed_{seed}.torch')))
                
                model = model.to(FLAGS.device)

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

            elif 'sEH' in FLAGS.exp[0] and len(FLAGS.exp) == 1: # triazine sEH FFNN
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
                
            elif 'SIRT2' in FLAGS.exp[0] and len(FLAGS.exp) == 1: # triazine SIRT2 FFNN
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

            logging.info(str(model))
            with open(LOG_FILE, 'a') as lf:
                lf.write(f'{datetime.now()} INFO: ' + str(model) + '\n')

            # Get MSE losses
            model.eval_metric = losses.loss_fn_MSE
            test_losses, test_enrichments = model.evaluate_on_del(x, exp_counts, bead_counts, test_slice, device=FLAGS.device)
            avg_test_loss = np.sum(test_losses, axis=0) / test_losses.shape[0]
            mse_losses.append(np.squeeze(avg_test_loss))
            formatted_loss = ['{0:.5f}'.format(loss) for loss in avg_test_loss]
            
            logging.info(f'MSE loss:         {np.squeeze(formatted_loss)}')    
            
            with open(LOG_FILE, 'a') as lf:      
                lf.write(f'{datetime.now()} INFO: MSE loss:         {np.squeeze(formatted_loss)}\n')
                                        
            # Record
            if not os.path.isfile(RESULTS_FILE):
                with open(RESULTS_FILE, 'w') as fid:
                    cols = ['Time (of recording)']
                    cols.append('Data split')
                    cols.append('Random seed')
                    cols.append('MSE loss')
                    fid.write('\t'.join(cols) + '\n')
            with open(RESULTS_FILE, 'a') as fid:
                cols = [str(datetime.now().now())]
                cols.append(str(splitter_names[i]))
                cols.append(str(seed))
                cols.append(str(np.squeeze(avg_test_loss)))
                fid.write('\t'.join(cols) + '\n')
            fid.close()
        
        # average across replicates
        avg_mse_loss = sum(mse_losses) / len(mse_losses)
        formatted_avg_loss = '{0:.5f}'.format(avg_mse_loss)
            
        logging.info(f'Average MSE loss: {np.squeeze(formatted_avg_loss)}\n')                     
        
        with open(LOG_FILE, 'a') as lf: 
            lf.write(f'{datetime.now()} INFO: Average MSE loss: {np.squeeze(formatted_avg_loss)}\n')
            
    lf.close()
    
if __name__ == '__main__':
    app.run(main)
    logging.shutdown()
 