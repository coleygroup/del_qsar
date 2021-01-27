import logging
from datetime import datetime
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from chemprop.data.data import MoleculeDataset, MoleculeDataLoader
from chemprop.models.mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights, NoamLR

# Using code from https://github.com/chemprop/chemprop for directed message passing neural networks

BATCH_SIZE = 1024
MAX_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 5

from del_qsar import utils, metrics

class DELQSARModel(nn.Module):
    def __init__(self):
        super(DELQSARModel, self).__init__()
        self.MPN = None
        self.loss_fn_train = None
        self.loss_fn_eval = None
        self.optimizer = None
        self.scheduler = None
        self.train_args = None
        self.sigmoid = None
        self.train_and_valid = None

    def train_on_del(self, x, exp_counts, bead_counts, 
            train_slice, valid_slice, true_labels=None, batch_size=BATCH_SIZE,
            num_workers=20, max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING_PATIENCE,
            zscale=lambda epoch: 1 + 9*np.exp(-epoch/2),
            reportfreq=1, max_norm=5, device=None, output_size=1, 
            save_path='best_model.torch', 
            log_path='run.log',
            torch_seed=None):
        
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            
        self.train_and_valid = True

        if self.loss_fn_train is None:
            raise ValueError('Model loss function undefined')
        if self.optimizer is None:
            raise ValueError('Model optimizer undefined')

        exp_tot = np.sum(exp_counts, axis=0)
        bead_tot = np.sum(bead_counts, axis=0)
        train_slice = train_slice.copy()
        
        self.all_train_losses = []
        self.all_valid_losses = []
        self.best_val_loss = np.inf
        for epoch in tqdm(range(max_epochs), desc='Training epochs'):
            logging.info(f'Starting epoch {epoch}')
            with open(log_path, 'a') as lf:
                lf.write(f'{datetime.now()} INFO: Starting epoch {epoch}\n')
            # Train for one epoch
            self.train()
            train_losses = []
            train_n = 0
            np.random.shuffle(train_slice)
            
            utils_train_batches = utils.batch(train_slice, batch_size, pad=True)
            
            if self.MPN:
                train_datapoints = []
                pad_fill_size = int(batch_size - (len(train_slice) % batch_size))
                train_slice_padded = np.append(train_slice, train_slice[:pad_fill_size])
                train_datapoints = [x[i] for i in train_slice_padded]
                train_data = MoleculeDataset(train_datapoints)
                mpn_train_batches = MoleculeDataLoader(
                    dataset=train_data,
                    batch_size=batch_size,
                    num_workers=num_workers
                ) 
                
                for batch, batch_indices in zip(mpn_train_batches, utils_train_batches):
                    # Prepare batch                
                    batch_x = batch.batch_graph()
                    features_batch = batch.features() 
                        
                    # Step
                    self.zero_grad()               
                    preds = self(batch_x, features_batch)
                    if exp_counts.shape[1] == 1:
                        preds = torch.unsqueeze(preds, 1)

                    losses = torch.zeros(batch_size, exp_counts.shape[1])
                    
                    if self.classification:
                        true_labels = true_labels.to(device)
                        for j in range(exp_counts.shape[1]): # iterating over POIs
                            losses_for_POI = self.loss_fn_train(preds[:, j], true_labels[batch_indices])
                            losses[:, j] = losses_for_POI
                    else:
                        for j in range(exp_counts.shape[1]): # iterating over POIs
                            k1 = torch.FloatTensor(exp_counts[batch_indices, j])
                            k2 = torch.FloatTensor(bead_counts[batch_indices, j])
                            n1 = float(exp_tot[j])
                            n2 = float(bead_tot[j])

                            if device:
                                k1 = k1.to(device)
                                k2 = k2.to(device)

                            losses_for_POI = self.loss_fn_train(preds[:, j], k1, k2, n1, n2, 
                                                    zscale=zscale(epoch))
                            losses[:, j] = losses_for_POI

                    losses_col_sum = torch.sum(losses, dim=0)
                    normalized_loss = losses.sum() / losses.shape[0]
                    normalized_loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
                    self.optimizer.step()
                
                    if isinstance(self.scheduler, NoamLR):
                        self.scheduler.step()
                
                    # Record
                    losses_col_sum = losses_col_sum.data.cpu().numpy()
                    losses = losses.data.cpu().numpy()
                    train_losses.append([loss.item() for loss in losses_col_sum])
                    train_n += len(batch_indices)
                
                    # Check for exploding loss
                    if np.isnan(losses.sum()):
                        self.best_val_loss = float('inf')
                        return self.best_val_loss
                        # raise ValueError('Loss is nan!')
              
            else: 
                for batch_indices in utils_train_batches:
                    # Prepare batch 
                    batch_x = torch.FloatTensor(x[batch_indices, :].astype(float))

                    if device:
                        batch_x = batch_x.to(device)
                        
                    # Step
                    self.optimizer.zero_grad()
                    preds = self(batch_x)
                    if exp_counts.shape[1] == 1:
                        preds = torch.unsqueeze(preds, 1)
                    
                    losses = torch.zeros(batch_size, exp_counts.shape[1])
                    if self.classification:
                        true_labels = true_labels.to(device)
                        for j in range(exp_counts.shape[1]): # iterating over POIs
                            losses_for_POI = self.loss_fn_train(preds[:, j], true_labels[batch_indices])
                            losses[:, j] = losses_for_POI
                    else:
                        for j in range(exp_counts.shape[1]): # iterating over POIs
                            k1 = torch.FloatTensor(exp_counts[batch_indices, j])
                            k2 = torch.FloatTensor(bead_counts[batch_indices, j])
                            n1 = float(exp_tot[j])
                            n2 = float(bead_tot[j])

                            if device:
                                k1 = k1.to(device)
                                k2 = k2.to(device)

                            losses_for_POI = self.loss_fn_train(preds[:, j], k1, k2, n1, n2, 
                                                          zscale=zscale(epoch))
                            losses[:, j] = losses_for_POI
                    
                    losses_col_sum = torch.sum(losses, dim=0)                
                    normalized_loss = losses.sum() / losses.shape[0]
                    normalized_loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
                    self.optimizer.step()
                
                    if isinstance(self.scheduler, NoamLR):
                        self.scheduler.step()
                
                    # Record    
                    losses_col_sum = losses_col_sum.data.cpu().numpy()
                    losses = losses.data.cpu().numpy()
                    train_losses.append([loss.item() for loss in losses_col_sum])
                    train_n += len(batch_indices)
                
                    # Check for exploding loss
                    if np.isnan(losses.sum()):
                        self.best_val_loss = float('inf')
                        return self.best_val_loss
                        # raise ValueError('Loss is nan!')
                                   
            # Report
            self.all_train_losses.append([sum(loss)/train_n for loss in zip(*train_losses)])
            if epoch % reportfreq == 0:
                formatted_loss = ['{0:8.4f}'.format(loss) for loss in self.all_train_losses[-1]]
                logging.info(f'Average training loss (scaled): {np.squeeze(formatted_loss)}')
                with open(log_path, 'a') as lf:
                    lf.write(f'{datetime.now()} INFO: Average training loss (scaled): {np.squeeze(formatted_loss)}\n')
                    
            # Evaluate on validation
            self.eval()
            valid_losses = []
            
            utils_valid_batches = utils.batch(valid_slice, batch_size)
            
            if self.MPN:
                valid_datapoints = [x[i] for i in valid_slice]
                valid_data = MoleculeDataset(valid_datapoints)
                mpn_valid_batches = MoleculeDataLoader(
                    dataset=valid_data,
                    batch_size=batch_size,
                    num_workers=num_workers
                )
           
                with torch.no_grad():                  
                    leftover_size = len(valid_slice) % batch_size
                    num_full_batches = (len(valid_slice) - leftover_size) / batch_size
                    batch_ctr = 0
                    for batch, batch_indices in zip(mpn_valid_batches, utils_valid_batches):
                        batch_ctr += 1
                        # Prepare batch
                        batch_x = batch.batch_graph()
                        features_batch = batch.features()                     

                        # Predict
                        preds = self(batch_x, features_batch) 
                        if len(batch_indices) == 1:
                            preds = torch.unsqueeze(preds, 0)
                        if exp_counts.shape[1] == 1:
                            preds = torch.unsqueeze(preds, 1) 
                        
                        if batch_ctr <= num_full_batches:
                            losses = torch.zeros(batch_size, exp_counts.shape[1]) 
                        else:
                            losses = torch.zeros(leftover_size, exp_counts.shape[1])
                            
                        if self.classification:
                            true_labels = true_labels.to(device)
                            for j in range(exp_counts.shape[1]): # iterating over POIs
                                losses_for_POI = self.loss_fn_train(preds[:, j], true_labels[batch_indices])
                                losses[:, j] = losses_for_POI
                        else:
                            for j in range(exp_counts.shape[1]): # iterating over POIs
                                k1 = torch.FloatTensor(exp_counts[batch_indices, j])
                                k2 = torch.FloatTensor(bead_counts[batch_indices, j])
                                n1 = float(exp_tot[j])
                                n2 = float(bead_tot[j])

                                if device:
                                    k1 = k1.to(device)
                                    k2 = k2.to(device)

                            losses_for_POI = self.loss_fn_train(preds[:, j], k1, k2, n1, n2)
                            losses[:, j] = losses_for_POI
                            
                        losses_col_sum = torch.sum(losses, dim=0)  
                        
                        # Record
                        losses_col_sum = losses_col_sum.data.cpu().numpy()
                        valid_losses.append([loss.item() for loss in losses_col_sum])
                        
            else:
                with torch.no_grad():
                    leftover_size = len(valid_slice) % batch_size
                    num_full_batches = (len(valid_slice) - leftover_size) / batch_size
                    batch_ctr = 0
                    for batch_indices in utils_valid_batches:
                        batch_ctr += 1
                        # Prepare batch
                        batch_x = torch.FloatTensor(x[batch_indices, :].astype(float))

                        if device: 
                            batch_x = batch_x.to(device)
                            
                        # Predict
                        preds = self(batch_x)
                        if len(batch_indices) == 1:
                            preds = torch.unsqueeze(preds, 0)
                        if exp_counts.shape[1] == 1:
                            preds = torch.unsqueeze(preds, 1)     
                        
                        if batch_ctr <= num_full_batches:
                            losses = torch.zeros(batch_size, exp_counts.shape[1]) 
                        else:
                            losses = torch.zeros(leftover_size, exp_counts.shape[1])
                            
                        if self.classification:
                            true_labels = true_labels.to(device)
                            for j in range(exp_counts.shape[1]): # iterating over POIs
                                losses_for_POI = self.loss_fn_train(preds[:, j], true_labels[batch_indices])
                                losses[:, j] = losses_for_POI
                        else:
                            for j in range(exp_counts.shape[1]): # iterating over POIs
                                k1 = torch.FloatTensor(exp_counts[batch_indices, j])
                                k2 = torch.FloatTensor(bead_counts[batch_indices, j])
                                n1 = float(exp_tot[j])
                                n2 = float(bead_tot[j])

                                if device:
                                    k1 = k1.to(device)
                                    k2 = k2.to(device)

                                losses_for_POI = self.loss_fn_train(preds[:, j], k1, k2, n1, n2)
                                losses[:, j] = losses_for_POI
                        
                        losses_col_sum = torch.sum(losses, dim=0)     
            
                        # Record
                        valid_losses.append([loss.item() for loss in losses_col_sum])
                        
            # Report
            self.all_valid_losses.append([sum(loss)/len(valid_slice) for loss in zip(*valid_losses)])
            if epoch % reportfreq == 0:
                formatted_loss = ['{0:8.4f}'.format(loss) for loss in self.all_valid_losses[-1]]
                logging.info(f'Average validation loss: {np.squeeze(formatted_loss)}')
                with open(log_path, 'a') as lf:
                    lf.write(f'{datetime.now()} INFO: Average validation loss: {np.squeeze(formatted_loss)}\n')
                            
            # Early stopping
            if sum(self.all_valid_losses[-1]) < self.best_val_loss:
                self.best_val_loss = sum(self.all_valid_losses[-1])
                torch.save(self.state_dict(), save_path)
            else:
                if all(sum(self.all_valid_losses[-i]) > self.best_val_loss for i in range(1, patience+1)):
                    logging.info(f'{patience} epochs without improving, stopping')
                    with open(log_path, 'a') as lf:
                        lf.write(f'{datetime.now()} INFO: {patience} epochs without improving, stopping\n')
                    break

        # Done training
        logging.info(f'Reloading best model state')
        with open(log_path, 'a') as lf:
            lf.write(f'{datetime.now()} INFO: Reloading best model state\n')
        self.load_state_dict(torch.load(save_path))
        return self.all_train_losses, self.all_valid_losses, self.best_val_loss

    def evaluate_on_del(self, x, exp_counts, bead_counts, test_slice, 
            batch_size=BATCH_SIZE, num_workers=20, device=None, true_labels=None):
        
        self.train_and_valid = False

        exp_tot = np.sum(exp_counts, axis=0)
        bead_tot = np.sum(bead_counts, axis=0)
        test_preds = []

        self.eval()
        if self.MPN: 
            if not self.classification:
                test_losses = []
            
            utils_test_batches = utils.batch(test_slice, batch_size)
            
            test_datapoints = [x[i] for i in test_slice]
            test_data = MoleculeDataset(test_datapoints)
            mpn_test_batches = MoleculeDataLoader(
                dataset=test_data,
                batch_size=batch_size,
                num_workers=num_workers
            )
        
            with torch.no_grad():
                leftover_size = len(test_slice) % batch_size
                num_full_batches = (len(test_slice) - leftover_size) / batch_size
                batch_ctr = 0
                for batch, batch_indices in tqdm(zip(mpn_test_batches, utils_test_batches)):
                    batch_ctr += 1
                    # Prepare batch
                    batch_x = batch.batch_graph()
                    features_batch = batch.features() 
                    
                    # Predict
                    preds = self(batch_x, features_batch)
                    if len(batch_indices) == 1:
                        preds = torch.unsqueeze(preds, 0)
                    if exp_counts.shape[1] == 1:
                        preds = torch.unsqueeze(preds, 1)      
                    for p in preds:
                        test_preds.append([_.item() for _ in p]) 
                    
                    if not self.classification:
                        if batch_ctr <= num_full_batches:
                            losses = torch.zeros(batch_size, exp_counts.shape[1]) 
                        else:
                            losses = torch.zeros(leftover_size, exp_counts.shape[1])
                        for j in range(exp_counts.shape[1]): # iterating over POIs
                            k1 = torch.FloatTensor(exp_counts[batch_indices, j])
                            k2 = torch.FloatTensor(bead_counts[batch_indices, j])
                            n1 = float(exp_tot[j])
                            n2 = float(bead_tot[j])

                            if device:
                                k1 = k1.to(device)
                                k2 = k2.to(device)

                            losses_for_POI = self.loss_fn_eval(preds[:, j], k1, k2, n1, n2)
                            losses[:, j] = losses_for_POI

                        # Record                  
                        for l in losses:
                            test_losses.append([_.item() for _ in l])
                            
                if self.classification:
                    test_roc_auc = np.zeros((1, exp_counts.shape[1]))
                    test_pr_auc = np.zeros((1, exp_counts.shape[1]))
                    for j in range(exp_counts.shape[1]): # iterating over POIs   
                        roc_auc_for_POI = metrics.get_roc_auc(true_labels[test_slice], test_preds)
                        pr_auc_for_POI = metrics.get_pr_auc(true_labels[test_slice], test_preds)
                        test_roc_auc[:, j] = roc_auc_for_POI
                        test_pr_auc[:, j] = pr_auc_for_POI
                    
        else: 
            if not self.classification:
                test_losses = []
            
            utils_test_batches = utils.batch(test_slice, batch_size)
            
            with torch.no_grad():
                leftover_size = len(test_slice) % batch_size
                num_full_batches = (len(test_slice) - leftover_size) / batch_size
                batch_ctr = 0
                for batch_indices in tqdm(utils_test_batches):
                    batch_ctr += 1
                    # Prepare batch
                    batch_x = torch.FloatTensor(x[batch_indices, :].astype(float))
 
                    if device:
                        batch_x = batch_x.to(device)
 
                    # Predict
                    preds = self(batch_x)
                    if len(batch_indices) == 1:
                        preds = torch.unsqueeze(preds, 0)
                    if exp_counts.shape[1] == 1:
                        preds = torch.unsqueeze(preds, 1)
                    for p in preds:
                        test_preds.append([_.item() for _ in p]) 
                        
                    if not self.classification:
                        if batch_ctr <= num_full_batches:
                            losses = torch.zeros(batch_size, exp_counts.shape[1]) 
                        else:
                            losses = torch.zeros(leftover_size, exp_counts.shape[1])
                        for j in range(exp_counts.shape[1]): # iterating over POIs
                            k1 = torch.FloatTensor(exp_counts[batch_indices, j])
                            k2 = torch.FloatTensor(bead_counts[batch_indices, j])
                            n1 = float(exp_tot[j])
                            n2 = float(bead_tot[j])

                            if device:
                                k1 = k1.to(device)
                                k2 = k2.to(device)

                            losses_for_POI = self.loss_fn_eval(preds[:, j], k1, k2, n1, n2)
                            losses[:, j] = losses_for_POI

                        # Record
                        for l in losses:
                            test_losses.append([_.item() for _ in l])
                           
                if self.classification:
                    test_roc_auc = np.zeros((1, exp_counts.shape[1]))
                    test_pr_auc = np.zeros((1, exp_counts.shape[1]))
                    for j in range(exp_counts.shape[1]): # iterating over POIs   
                        roc_auc_for_POI = metrics.get_roc_auc(true_labels[test_slice], test_preds)
                        pr_auc_for_POI = metrics.get_pr_auc(true_labels[test_slice], test_preds)
                        test_roc_auc[:, j] = roc_auc_for_POI
                        test_pr_auc[:, j] = pr_auc_for_POI
                    
        # Convert
        test_preds = np.array(test_preds, dtype=float)
        if self.classification:
            # test_roc_auc = np.array(test_roc_auc, dtype=float)
            # test_pr_auc = np.array(test_pr_auc, dtype=float)
            return test_roc_auc, test_pr_auc, test_preds
        else:
            test_losses = np.array(test_losses, dtype=float)
            return test_losses, test_preds

    def predict_on_x(self, x_predict, batch_size=BATCH_SIZE, num_workers=20, device=None):
        if self.MPN:
            num_compounds = len(x_predict)
        else:
            num_compounds = x_predict.shape[0]
            
        predict_slice = np.arange(num_compounds)
        self.eval()
        
        if not self.MPN and x_predict.ndim == 1: # just one sample
            single_x = torch.FloatTensor(np.expand_dims(x_predict, 0))
            if device:
                single_x = single_x.to(device)
            return float(self(single_x))
        if self.MPN and len(x_predict) == 1:
            return float(self(
                MoleculeDataset(x_predict).batch_graph(),
                MoleculeDataset(x_predict).features() 
            ))
            
        all_preds = []
        
        utils_batches = utils.batch(predict_slice, batch_size)
        
        if self.MPN:
            test_datapoints = [x_predict[i] for i in predict_slice]
            test_data = MoleculeDataset(test_datapoints)
            mpn_batches = MoleculeDataLoader(
                dataset=test_data,
                batch_size=batch_size,
                num_workers=num_workers
            )

            with torch.no_grad():
                for batch, batch_indices in tqdm(zip(mpn_batches, utils_batches)):
                    batch_x = batch.batch_graph()
                    features_batch = batch.features() 
                    preds = self(batch_x, features_batch) 
                    if len(batch_indices) == 1:
                        preds = torch.unsqueeze(preds, 0)
                    preds = preds.data.cpu().numpy()
                    all_preds.extend(list(preds))                
            return np.array(all_preds)
        
        else:
            with torch.no_grad():
                for batch_indices in utils_batches:
                    batch_x = torch.FloatTensor(x_predict[batch_indices, :])
                    if device:
                        batch_x = batch_x.to(device)
                    preds = self(batch_x)
                    if len(batch_indices) == 1:
                        preds = torch.unsqueeze(preds, 0)
                    preds = preds.data.cpu().numpy()
                    all_preds.extend(list(preds))
            return np.array(all_preds)
        
        
class MLP(DELQSARModel):
    def __init__(self, input_size, layer_sizes, dropout=0.2, num_tasks=1, task_type='regression', torch_seed=None):
        super(MLP, self).__init__()
        
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        
        self.classification = task_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.MPN = False
            
        layers = [nn.Linear(input_size, layer_sizes[0])]
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        layers.append(nn.Linear(layer_sizes[-1], num_tasks))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        if not self.classification:
            return F.softplus(torch.squeeze(self.layers(x))) # Use special activation for enrichment
        elif not self.train_and_valid:
            return self.sigmoid(torch.squeeze(self.layers(x)))
        else:
            return torch.squeeze(self.layers(x))
            

class MoleculeModel(DELQSARModel):
    """Directed message passing network followed by feed-forward layers"""
    def __init__(self, featurizer = False,
                 dataset_type = 'regression', 
                 num_tasks = 1,
                 atom_messages = False,
                 bias = False, 
                 init_lr = 1e-4,
                 max_lr = 1e-3,
                 final_lr = 1e-4,
                 depth = 3, 
                 dropout = 0.0,
                 undirected = False, 
                 features_only = False,
                 use_input_features = False,
                 features_size = None,
                 activation = 'ReLU',
                 hidden_size = 300, 
                 ffn_hidden_size = None,
                 ffn_num_layers = 2,
                 device = 'cuda:0',
                 torch_seed=None):
        super(MoleculeModel, self).__init__()
        
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        
        self.featurizer = featurizer
        self.classification = dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.MPN = True

        self.output_size = num_tasks
        self.device = device

        self.create_encoder(
            atom_messages=atom_messages, hidden_size=hidden_size,
            bias=bias, depth=depth, dropout=dropout,
            undirected=undirected, features_only=features_only,
            use_input_features=use_input_features,
            features_size=features_size, activation=activation,
            device=self.device)
        self.create_ffn(
            output_size=self.output_size, features_only=features_only,
            features_size=features_size, hidden_size=hidden_size,
            use_input_features=use_input_features,
            dropout=dropout, activation=activation,
            ffn_num_layers=ffn_num_layers, ffn_hidden_size=ffn_hidden_size,
            device=self.device)

        initialize_weights(self)

    def create_encoder(self, atom_messages = False,
                       bias = False,
                       hidden_size = 300, 
                       depth = 3,
                       dropout = 0.0, 
                       undirected = False,
                       features_only = False,
                       use_input_features = False,
                       features_size = None,
                       activation = 'ReLU', 
                       device = 'cuda:0'):
        self.encoder = MPN(Namespace(
            atom_messages=atom_messages, hidden_size=hidden_size,
            bias=bias, depth=depth, dropout=dropout, undirected=undirected,
            features_only=features_only, use_input_features=use_input_features,
            features_size=features_size, activation=activation,
            device=device))

    def create_ffn(self, output_size, 
                   features_only = False,
                   features_size = None,
                   hidden_size = 300,
                   use_input_features = False, 
                   dropout = 0.0,
                   activation = 'ReLU', 
                   ffn_num_layers = 2,
                   ffn_hidden_size = None, # ffn_hidden_size defaults to hidden_size
                   device = 'cuda:0') -> None: 
        first_linear_dim = hidden_size

        dropout = nn.Dropout(dropout)
        activation = get_activation_function(activation)
        
        # Create FFN layers
        if ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            if ffn_hidden_size is None:
                ffn_hidden_size = hidden_size

            ffn = [
                dropout,
                nn.Linear(first_linear_dim, ffn_hidden_size)
            ]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(ffn_hidden_size, ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(ffn_hidden_size, output_size),
            ])
        
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.ffn = self.ffn.to(device)

    def forward(self, *input):
        if self.featurizer:
            return self.featurize(*input)
        
        if not self.classification:
            return F.softplus(torch.squeeze(self.ffn(self.encoder(*input))))
        elif not self.train_and_valid: # For binary classifier: don't apply sigmoid during training b/c using BCEWithLogitsLoss
            return self.sigmoid(torch.squeeze(self.ffn(self.encoder(*input))))
        else:
            return torch.squeeze(self.ffn(self.encoder(*input)))
