# DEL QSAR
Prediction of enrichment from molecular structure, using DNA-encoded libraries and machine learning with a probabilistic loss function

## Dependencies
- Python (3.6.10)
- [chemprop](https://github.com/chemprop/chemprop) (0.0.2)
- Numpy (1.18.5)
- pandas (1.0.4)
- PyTorch (1.5.0)
- RDKit (2020.03.2)
- scikit-learn (0.23.1)
- tqdm (4.46.1)
- h5py (2.10.0; for writing/reading an HDF5 file with stored fingerprints)
- [Optuna](https://github.com/optuna/optuna) (1.5.0; for hyperparameter optimization)
- hickle (3.4.5; for saving/loading dictionaries in the bit and substructure analysis)

## Instructions

### Initial set-up

- Unzip `triazine_lib_sEH_SIRT2_QSAR.csv.gz` in the `experiments/datasets` folder
- Unzip `pubchem_smiles.npy.gz` in the `experiments/visualizations` folder
- To generate and save molecular fingerprints to an HDF5 file, navigate to the `experiments` folder and run:
```
python fps_preprocessing.py --csv </path/to/data.csv from experiments/datasets folder> --fps_h5 <filename for HDF5 file to create> --fp_size <fingerprint size (number of bits)>
```

### Training and evaluating models

To train/evaluate models, navigate to the `experiments` folder.
The following is a general command for running one of the scripts:
```
python <script_name.py> --csv </path/to/data.csv from experiments/datasets folder> --out <experiment label (name of results subfolder to create)> --device <device (set to cuda:0 if using GPU)>
```
Additional command-line arguments include:
- `--exp <column header(s) for barcode counts in experiment with protein(s) of interest>`
  - Replace spaces with underscores
- `--beads <column header(s) for barcode counts in beads-only control without protein(s) of interest>`
  - Specify beads column header for each protein of interest, in the same order as listed in `--exp`
  - Replace spaces with underscores
  
The script depends on the dataset/model type/task. Further details are provided below, including more specific command-line arguments.

#### Regression models and binary classifiers:

Scripts:
- General:
  - `single_model_run.py` (training and evaluating a model with user-provided hyperparameter values)
- For DD1S CAIX dataset:
  - `hyperparameter_optimization_DD1S_CAIX_FFN.py` (optimizing feed-forward neural networks)
  - `hyperparameter_optimization_DD1S_CAIX_MPN.py` (optimizing directed message passing neural networks)
- For triazine sEH and SIRT2 datasets:
  - `hyperparameter_optimization_triazine_FFN.py` (optimizing feed-forward neural networks)
  - `triazine_MPN_LR_tuning.py` (tuning learning rate for directed message passing neural networks)
  - `triazine_MPN.py` (training and evaluating directed message passing neural networks)

Additional command-line arguments include:
- `--featurizer <type of molecule featurization>`
  - For feed-forward neural networks: `'fingerprint'` or `'onehot'`
  - For directed message passing neural networks: `'graph'`
- `--splitter <type of data split for train/validation/test>`
  - Options include:
  - `'random'`
  - any combination of `'cycle1'`, `'cycle2'`, `'cycle3'` (for example: `'cycle1','cycle3'`)
  - To reproduce results for cycle splits along multiple cycles, list the cycles in ascending numerical order
- `--seed <random seed for data splitting and weight initialization>`
- `--task_type <task type>`
  - Options include:
  - `'regression'`
  - `'classification'`
- `--loss_fn_train <loss function to use during training>`
  - For binary classifiers: `'BCE'` 
  - Options for regression models include: 
  - `'nlogprob'` (probabilistic loss function)
  - `'MSE'` (mean squared error)
- `--max_epochs <maximum number of epochs>`
  - Values for reproducing results:
  - Training feed-forward neural networks: 1000 for DD1S CAIX dataset; 300 for triazine sEH and SIRT2 datasets
  - Training directed message passing neural networks: 100
- `--patience <patience>`
  - To reproduce results, set to 5
- `--max_norm <max norm>`
  - To reproduce results, set to 5

For fingerprint featurization (if reading from an HDF5 file with stored fingerprints):
- `--fps_h5 </path/to/file_with_stored_fingerprints.h5 from experiments folder>`

For hyperparameter optimization (including `triazine_MPN_LR_tuning.py`):
- `--n_trials <number of trials to run/hyperparameter sets to try>`
  - Values for reproducing results:
  - For the DD1S CAIX dataset: 100
  - For the triazine sEH and SIRT2 datasets: 25 for feed-forward neural networks, 4 for directed message passing neural networks

For `single_model_run.py`:
- `--model_type <model type>`
  - Options include:
  - `'MLP'` (feed-forward neural network)
  - `'MoleculeModel'` (directed message passing neural network)
- `--lr <initial learning rate>`
- `--dropout <dropout rate>`
- `--eval_metric <loss function used to evaluate the model>`
  - Options include:
  - `'NLL'` (negative log-likelihood)
  - `'MSE'` (mean-squared error)
- For feed-forward neural network:
  - `--layer_sizes <hidden layer sizes>`
- For directed message passing neural network:
  - `--depth <number of message-passing steps>`
  - `--hidden_size <size of hidden layers>`
  - `--ffn_num_layers <number of feed-forward network layers>`

For directed message passing neural networks:
- `--num_workers <number of workers for loading data>`

For directed message passing neural networks specifically on the triazine sEH and SIRT2 datasets:
- `--depth <number of message-passing steps>`
- `--hidden_size <size of hidden layers>`
- `--ffn_num_layers <number of feed-forward network layers>`

For `triazine_MPN.py`:
- `--lr <initial learning rate>`
- `--dropout <dropout rate>`

For binary classifiers:
- `--threshold_type <type of threshold for determining ground truth labels>`
  - Options include:
  - `'percentile'`
  - `'fixed'`
  - To reproduce results, set to `'percentile'`
- `--threshold_val <threshold value; percentile or exact value>`
  - Percentile example: `99.5` to define the top 0.5% of training set compounds as enriched

#### Evaluating trained regression models as binary classifiers, or using MSE loss/rank correlation coefficient

Scripts:
- `bin_eval.py` (fixed threshold)
- `bin_eval_multiple_thresholds.py` (multiple thresholds)
- `mse_loss_eval.py`
- `rank_corr_coeff_eval.py`
  
These scripts require 
(1) an `experiments/models` folder with saved regression models (`.torch` files) organized by dataset/model type and named by data split/seed, as follows (for brevity, only filenames for the random splits are shown; cycle-split models should also be included, replacing `random` with `cycle1`, ..., `cycle12`, ..., `cycle123` in the filename):
```
└── models
    ├── DD1S_CAIX
    │   └── D-MPNN
    │   │   └── random_seed_0.torch 
    │   │   └── random_seed_1.torch
    │   │   └── random_seed_2.torch
    │   │   └── random_seed_3.torch
    │   │   └── random_seed_4.torch
    │   │   
    │   └── D-MPNN_pt
    │   │   └── (same as for models/DD1S_CAIX/D-MPNN)
    │   │
    │   └── FP-FFNN
    │   │   └── (same as for models/DD1S_CAIX/D-MPNN)
    │   │
    │   └── FP-FFNN_pt
    │   │   └── (same as for models/DD1S_CAIX/D-MPNN)
    │   │
    │   └── OH-FFNN
    │   │   └── (same as for models/DD1S_CAIX/D-MPNN)
    │   │
    │   └── OH-FFNN_pt
    │       └── (same as for models/DD1S_CAIX/D-MPNN)
    │    
    ├── triazine_sEH
    │   └── D-MPNN
    │   │   └── random_seed_0.torch
    │   │   └── random_seed_1.torch
    │   │   └── random_seed_2.torch
    │   │ 
    │   └── D-MPNN_pt
    │   │   └── (same as for models/triazine_sEH/D-MPNN)
    │   │
    │   └── FP-FFNN
    │   │   └── random_seed_0.torch
    │   │   └── random_seed_1.torch
    │   │   └── random_seed_2.torch
    │   │   └── random_seed_3.torch
    │   │   └── random_seed_4.torch
    │   │   
    │   └── FP-FFNN_pt
    │   │   └── (same as for models/triazine_sEH/FP-FFNN)
    │   │  
    │   └── OH-FFNN
    │   │   └── (same as for models/triazine_sEH/FP-FFNN)
    │   │  
    │   └── OH-FFNN_pt
    │       └── (same as for models/triazine_sEH/FP-FFNN)
    │
    ├── triazine_SIRT2
    │   └── (same as for models/triazine_sEH)
    │
    └── triazine_sEH_SIRT2_multi-task
        └── D-MPNN
        │   └── random_seed_0.torch
        │   └── random_seed_1.torch
        │   └── random_seed_2.torch
        │
        └── FP-FFNN
        │   └── random_seed_0.torch
        │   └── random_seed_1.torch
        │   └── random_seed_2.torch
        │   └── random_seed_3.torch
        │   └── random_seed_4.torch
        │
        └── OH-FFNN
            └── (same as for models/triazine_sEH_SIRT2_multi-task/FP-FFNN)

```
(2) a csv with the hyperparameter values of the saved models, formatted like the following (example values are shown):

| dataset        | model type | seed | split  | layer sizes   | dropout | depth | hidden size | FFN num layers |
|----------------|------------|------|--------|---------------|---------|-------|-------------|----------------|
| DD1S_CAIX      | D-MPNN     | 0    | random |               | 0.1     | 6     | 1300        | 3              |
| triazine_sEH   | FP-FFNN_pt | 1    | random | 128, 128      | 0.35    |       |             |                |
| triazine_SIRT2 | OH-FFNN    | 1    | random | 512, 256, 128 | 0.45    |       |             |                |

For each of the above scripts, each run evaluates all regression models (only on random splits for evaluating models as binary classifiers) for a given dataset and model type. `bin_eval.py` and `bin_eval_multiple_thresholds.py` evaluate the models as binary classifiers; `mse_loss_eval.py` evaluates models using MSE loss; `rank_corr_coeff_eval.py` evaluates models using Spearman's rank correlation coefficient.

Additional command-line arguments include:
- `--model_type <model type>`
  - Options include:
  - `'D-MPNN'`
  - `'D-MPNN_pt'`
  - `'FP-FFNN'`
  - `'FP-FFNN_pt'`
  - `'OH-FFNN'`
  - `'OH-FFNN_pt'`
- `--hyperparams </path/to/hyperparameter_values_of_saved_models.csv from experiments folder>`

For `D-MPNN` and `D-MPNN_pt` models:
- `--num_workers <number of workers for loading data>`

Specifically for `bin_eval.py` and `bin_eval_multiple_thresholds.py`:
- `--random_split_only <True or False>`
  - Whether to evaluate the models for random split only
  - To reproduce results, set to `True`
- `--random_guess <True or False>`
  - Whether to generate random-guess baseline (instead of evaluating saved models)
  - To reproduce results, set `--model_type` to `'FP-FFNN'` when generating random-guess baseline
- `--threshold_type <type of threshold for determining ground truth labels>`
  - Options include:
  - `'percentile'`
  - `'fixed'`
  - To reproduce results, set to `'percentile'`
- `--threshold_val <threshold value; percentile or exact value>`
  - Percentile example: `99.5` to define the top 0.5% of training set compounds as enriched

Specifically for `bin_eval_multiple_thresholds.py`:
- `--num_thresholds <number of (logarithmically spaced) thresholds to try>`
  - To reproduce results, set to `20`
- `--start_idx <threshold index, from 0 to 19, to start at>`
  - Can set to `1` or higher if resuming a job
- `--stop_idx <threshold index, from 1 to 20, to stop at>`
  - Can set to `19` or lower if stopping early

#### KNN and random baselines

(1) To train and evaluate baseline k-nearest-neighbors (KNN) regression models, run the following scripts for each dataset:
- For the DD1S CAIX dataset: 
  - Run `knn_train_DD1S_CAIX.py`, specifying the type of molecule featurization (`--featurizer <'onehot' or 'fingerprint'>`) and the number of neighbors (`--n_neighbors <number of neighbors; values of 1, 3, 5, 7, 9 were tested>`). The script iterates through all data splits and random seeds, and saves the trained models in the results folder.
  - Create the folders `experiments/models/DD1S_CAIX/OH-KNN` (for the onehot KNNs) and `experiments/models/DD1S_CAIX/FP-KNN` (for the fingerprint-based KNNs), with subfolders `k_1`, `k_3`, `k_5`, `k_7`, `k_9` (for the different tested values of `n_neighbors`); move the corresponding saved models (`.joblib` files) to these folders. The filename for each saved model should be of the form `<data split>_seed_<random seed>.joblib`, where the possible data split names are`random`, `cycle1`, ..., `cycle12`, ..., `cycle123`.
  - Run `DD1S_CAIX_knn_eval.py`, specifying the metric used for evaluation (`--eval_metric <'NLL', 'MSE', or 'rank_corr_coeff'>`), the type of featurization (`--featurizer <'onehot' or 'fingerprint'>`), the data split (`--splitter`), and random seed (`--seed`). The script iterates through all tested `n_neighbors` values of 1, 3, 5, 7, 9.
- For the triazine datasets:
  - Run `knn_train_triazine.py`, specifying the type of molecule featurization (`--featurizer <'onehot' or 'fingerprint'>`), data split (`--splitter`), and random seed (`--seed`). To reproduce results, keep the default value of `9` for `--n_neighbors`. The script saves the trained model in the results folder.
  - Create the folders `experiments/models/<dataset>/OH-KNN/k_9` and `experiments/models/<dataset>/FP-KNN/k_9` (where `<dataset>` is `triazine_sEH` or `triazine_SIRT2`, and the subfolder `k_9` refers to the fixed value of 9 for `n_neighbors`); move the corresponding saved models (`.joblib` files) to these folders. KNNs were trained only for random seed 0 on the triazine datasets; the filename for each saved model should be of the form `<data split>_seed_0.joblib`.
  - Run `triazine_knn_generate_test_preds.py` to generate and save test-set predictions, specifying the metric used for evaluation (`--eval_metric <'NLL', 'MSE', or 'rank_corr_coeff'>`), the type of featurization (`--featurizer <'onehot' or 'fingerprint'>`), the data split (`--splitter`), and random seed (`--seed`). The script uses `9` as the value for `n_neighbors`. The generated test-set predictions are saved in the results folder.
  - Run `triazine_knn_eval_on_test_preds.py`, specifying the path to the saved test-set predictions (from the `/experiments/results` folder), the metric used for evaluation (`--eval_metric <'NLL', 'MSE', or 'rank_corr_coeff'>`), and the data split (`--splitter`).
    
(2) To run random baselines, run the script `random_baseline.py`, specifying via `--hyperparams` the filename of a csv (in the `experiments` folder) with the hyperparameter values of the saved models (for the specific format of the hyperparameters file, see (2) under "Evaluating trained regression models as binary classifiers, or using MSE loss/rank correlation coefficient"). Also specify the data split (`--splitter`), the type of random baseline (`--random_type <'shuffle_preds' or 'predict_all_ones'>`), and the metric used for model evaluation (`--eval_metric <'NLL', 'MSE', or 'rank_corr_coeff'>`).

### Visualizations

Scripts and notebooks for visualizations can be found in the `experiments/visualizations` folder

#### Atom-centered Gaussian visualizations for fingerprint-based models

To visualize atomic contributions to the predictions of a trained fingerprint-based model, run:
```
python visualize_smis.py --model_path </path/to/saved_model.torch from experiments folder> --cpd_ids <compound ID(s) of the compound(s) to visualize> --csv </path/to/data.csv from experiments/datasets folder> --fps_h5 </path/to/file_with_stored_fingerprints.h5 from experiments folder> --out <label for results subfolder> --layer_sizes <hidden layer sizes> --dropout <dropout rate>
```
where `--layer_sizes` and `--dropout` are hyperparameters of the saved model

#### Bit and substructure analysis

This visualization requires:
- (i) a file with stored fingerprints for the DD1S CAIX dataset (for instructions, see the "Initial set-up" section above)
- (ii) a file with stored fingerprints for the triazine library (for instructions, see the "Initial set-up" section above)
- (iii) saved random split FP-FFNN models; model hyperparameter values (including learning rate) are specified in `compiled_results.xlsx` in the `experiments/paper_results` folder
  - DD1S CAIX (seeds 0, 1, 2)
  - triazine sEH (seed 0)
  - triazine SIRT2 (seed 0)
  
To run the visualization, run all cells in `Single substructure analysis.ipynb` and `Substructure pair analysis.ipynb` (note: the substructure pair analysis depends on results from running the single substructure analysis). 

Fingerprint and model filenames/paths, etc. may be modified as necessary in the fourth cell from the top in each of the Jupyter notebooks. Information about the bits and substructures is printed out in the notebook; visualizations of the substructures are saved to png files, along with histograms and bar graphs. 

For each dataset in the single substructure analysis, a command-line alternative to running the first two cells under "Get and visualize substructures" is to run:
```
python single_substructure_analysis_get_substructures.py --csv </path/to/data.csv from experiments/datasets folder> --fps_h5 </path/to/file_with_stored_fingerprints.h5 from experiments folder> --dataset_label <'DD1S_CAIX', 'triazine_sEH', or 'triazine_SIRT2'> --seed <random seed for data splitting and weight initialization (only used for file naming in this context)> --bits_of_interest <bits of interest (top 5 and bottom 3 bits)>
```
After running this script, the rest of the single substructure analysis for the given dataset can be resumed in the Jupyter notebook, starting with the third cell (currently commented-out) under "Get and visualize substructures."

#### UMAP

1. Navigate to the `experiments/visualizations` folder
2. Generate 4096-bit fingerprints for the PubChem compounds (see `pubchem_smiles.npy` for the compounds' SMILES strings) by running `python generate_pubchem_fps.py`. Also generate 4096-bit fingerprints for DOS-DEL-1 and the triazine library, using the script `fps_preprocessing.py` in the `experiments` folder (for instructions, see the "Initial set-up" section above). The resulting files with stored fingerprints should be moved to the `experiments/visualizations` folder.
3. To train and apply a UMAP embedding, run:
```
python UMAP.py --num_threads <number of threads> --pubchem_fps_h5 <name of HDF5 file with stored 4096-bit fingerprints for PubChem> --DD1S_fps_h5 <name of HDF5 file with stored 4096-bit fingerprints for DOS-DEL-1> --triazine_fps_h5 <name of HDF5 file with stored 4096-bit fingerprints for the triazine library>
```
4. To generate UMAP plots, run all cells in `UMAP plots.ipynb`

#### Various plots

- Loss function plots
  - Run all cells in `Loss function plots.ipynb`
- Test performance bar graphs/scatter plots
  - Requires csv files `DD1S_CAIX_NLL_test_losses.csv`, `DD1S_CAIX_MSE_test_losses.csv`, `DD1S_CAIX_rank_corr_coeffs.csv`, `triazine_sEH_NLL_test_losses.csv`, `triazine_sEH_MSE_test_losses.csv`, `triazine_sEH_rank_corr_coeffs.csv`, `triazine_SIRT2_NLL_test_losses.csv`, `triazine_SIRT2_MSE_test_losses.csv`, `triazine_SIRT2_rank_corr_coeffs.csv`, `triazine_multitask_sEH_test_losses.csv`, and `triazine_multitask_SIRT2_test_losses.csv` in the `experiments` folder 
  - For the single-task model NLL/MSE/rank correlation coefficient csv files, the column headers should be `model type`, (`seed`), `split` (data split names recorded as `random`, `cycle1`, ..., `cycle12`, ..., `cycle123`), and `test performance` (loss or rank correlation coefficient on the test set). The model type names should be recorded as `OH-FFNN`, `OH-FFNN_pt`, `FP-FFNN`, `FP-FFNN_pt`, `D-MPNN`, and `D-MPNN_pt`
  - For the multi-task model test loss csv files (which should include test losses for both the multi-task models and the corresponding single-task models), the column headers should be `model type`, (`seed`), `split` (data split names recorded as `random`, `cycle123`), and `test performance` (loss on the test set). The model type names should be recorded as `OH-FFNN_single-task`, `FP-FFNN_single-task`, `D-MPNN_single-task`, `OH-FFNN_multi-task`, `FP-FFNN_multi-task`, and `D-MPNN_multi-task`
  - Run all cells in `Test performance bar graphs and scatter plots.ipynb`
- DD1S CAIX hyperparameter optimization result histograms
  - Requires two csv files `DD1S_CAIX_hyperparameter_optimization_results.csv` (for the D-MPNN / D-MPNN_pt regression models) and `bin_DD1S_CAIX_hyperparameter_optimization_results.csv` (for the random split D-MPNN binary classifiers) in the `experiments` folder, each formatted like the following (example values are shown):
  
    | model type | seed | split  | depth | FFN num layers | hidden size | dropout |
    |------------|------|--------|-------|----------------|-------------|---------|
    | D-MPNN     | 0    | random | 6     | 3              | 1300        | 0.10    |

  - Run all cells in `DD1S CAIX hyperparameter optimization result histograms.ipynb` in the `experiments` folder 
- DD1S CAIX KNN n_neighbors optimization result histograms
  - Requires a csv file `DD1S_CAIX_KNN_k_optimization_results.csv` in the `experiments` folder, with column headers `model type` (`OH-KNN` or `FP-KNN`), `seed` (`0`, `1`, `2`, `3`, or `4`), `metric` (`NLL`, `MSE`, or `rank corr coeff`), `split` (`random`, `cycle1`, ..., `cycle12`, ..., `cycle123`), and `k` (optimized number of neighbors)
  - Run all cells in `DD1S CAIX KNN k optimization result histograms.ipynb`
- DD1S CAIX histograms (1D and 2D) and parity plots; includes plots for both NLL- and MSE-trained models
  -  Requires:
  - (i) a file with stored fingerprints for the DD1S CAIX dataset (for instructions, see the "Initial set-up" section above)
  - (ii) two NLL-trained FP-FFNN models (random split, cycle 1+2 split) and one MSE-trained FP-FFNN (random split); model hyperparameter values (including learning rate) are specified in `compiled_results.xlsx` in the `experiments/paper_results` folder
  - Run all cells in `DD1S CAIX histograms and parity plots.ipynb` (can modify fingerprint and model filenames/paths as necessary, in the fourth cell from the top)
- DD1S CAIX disynthon plots
  -  Requires:
  - (i) a file with stored fingerprints for the DD1S CAIX dataset (for instructions, see the "Initial set-up" section above)
  - (ii) a random split FP-FFNN model; model hyperparameter values (including learning rate) are specified in `compiled_results.xlsx` in the `experiments/paper_results` folder
  - Run all cells in `DD1S CAIX disynthon parity plots and 1D histograms.ipynb` (can modify fingerprint and model filenames/paths as necessary, in the fourth cell from the top)
- Triazine sEH / SIRT2 parity plots
  - Requires:
  - (i) a file with stored fingerprints for the triazine library (for instructions, see the "Initial set-up" section above)
  - (ii) two saved random split FP-FFNN models (triazine sEH, triazine SIRT2); model hyperparameter values (including learning rate) are specified in `compiled_results.xlsx` in the `experiments/paper_results` folder
  - Run all cells in `Triazine parity plots.ipynb` (can modify fingerprint and model filenames/paths as necessary, in the fourth cell from the top)
- Triazine sEH / SIRT2 disynthon parity plots
  - Requires:
  - (i) a file with stored fingerprints for the triazine library (for instructions, see the "Initial set-up" section above)
  - (ii) two saved random split FP-FFNN models (triazine sEH, triazine SIRT2); model hyperparameter values (including learning rate) are specified in `compiled_results.xlsx` in the `experiments/paper_results` folder
  - Run all cells in `Triazine disynthon parity plots.ipynb` (can modify fingerprint and model filenames/paths as necessary, in the fourth cell from the top)
- Distributional shift histograms
  - Run all cells in `DD1S CAIX cycle 2 distributional shift.ipynb`
- Binary classification baseline comparison bar graphs (fixed threshold)
  - Requires a csv file `bin_AUCs.csv` in the `experiments` folder. Column headers should be `dataset`, `model type`, `seed`, `PR AUC`, and `ROC AUC`; datasets should be recorded as `DD1S CAIX`, `triazine sEH`, `triazine SIRT2`; model types should be recorded as `OH-FFNN`, `OH-FFNN pt`, `OH-FFNN bin`, `FP-FFNN`, `FP-FFNN pt`, `FP-FFNN bin`, `D-MPNN`, `D-MPNN pt`, `D-MPNN bin`, `Random guess`
  - Run all cells in `Fixed threshold bin plots.ipynb`
- Binary classification baseline comparison line graphs (multiple thresholds)
  - Requires a csv file `AUCs_multiple_thresholds.csv` in the `experiments` folder. Column headers should be `dataset`, `model type`, `top percent`, `seed`, `PR AUC`, and `ROC AUC`; datasets should be recorded as `DD1S CAIX`, `triazine sEH`, `triazine SIRT2`; model types should be recorded as `OH-FFNN`, `OH-FFNN pt`, `FP-FFNN`, `FP-FFNN pt`, `D-MPNN`, `D-MPNN pt`, `Random guess`; a `top percent` value of `x` (for example) means that the top `x`% of training set compounds are defined as enriched.
  - Run all cells in `Multiple thresholds bin plots.ipynb`
- 2D histograms of predicted vs. calculated enrichment for triazine sEH and SIRT2 datasets
  -  Requires:
  - (i) a file with stored fingerprints for the triazine library (for instructions, see the "Initial set-up" section above)
  - (ii) four saved FP-FFNN models (triazine sEH + random split, triazine sEH + cycle 1+2+3 split, triazine SIRT2 + random split, triazine SIRT2 + cycle 1+2+3 split); model hyperparameter values (including learning rate) are specified in `compiled_results.xlsx` in the `experiments/paper_results` folder
  - Run all cells in `Triazine 2D histograms.ipynb` (can modify fingerprint and model filenames/paths as necessary, in the fourth cell from the top)
- Parity plots to evaluate generalization ability
  -  Requires:
  - (i) a file with stored fingerprints for the triazine library (for instructions, see the "Initial Set-up" section above)
  - (ii) saved cycle 1+2+3 models (one for each (triazine dataset, model type, replicate/seed) triple, where the included model types are FP-FFNN, OH-FFNN, and D-MPNN); model hyperparameter values (including learning rate) are specified in `compiled_results.xlsx` in the `experiments/paper_results` folder
  - Run all cells in `Triazine generalization parity plots.ipynb` (can modify fingerprint and model filenames/paths as necessary, in the fourth cell from the top)
    
#### Outliers for the DD1S CAIX dataset

1) To train a FP-KNN on the entire dataset, run `FP-KNN_train_on_all_DD1S_CAIX.py` (specify a filename for the results folder with `--out`; otherwise, leave the command-line arguments as the default values). The script saves the trained model in a subfolder of the results folder.
2) To walk through the process used to identify example outliers in the DD1S CAIX dataset, see `Identifying DD1S CAIX outliers.ipynb` in the `experiments` folder
3) Run `DD1S_CAIX_outliers_get_nearest_neighbor.py` (use `--model_path` to specify the path, from the `experiments` folder, to the saved FP-KNN trained on the entire DD1S CAIX dataset) to obtain the index and SMILES string of the nearest neighbor in the DD1S CAIX dataset for each example outlier
