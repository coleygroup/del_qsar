import numpy as np

DATA_SPLIT_SEED = 0
TRAIN_FRAC = 0.7
TRAINVAL_FRAC = 0.8

class Splitter(object):
    pass


class RandomSplitter(Splitter):
    def __init__(self):
        pass

    def __call__(self, x, df_data, train_frac=TRAIN_FRAC, trainval_frac=TRAINVAL_FRAC,
                 seed=DATA_SPLIT_SEED):
        # Random
        data_indices = np.arange(int(len(df_data)))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(data_indices)

        train_slice = data_indices[:int(len(df_data) * train_frac)]
        valid_slice = data_indices[int(
            len(df_data) * train_frac):int(len(df_data) * trainval_frac)]
        test_slice = data_indices[int(len(df_data) * trainval_frac):]

        return train_slice, valid_slice, test_slice


class OneCycleSplitter(Splitter):
    def __init__(self, cycle_num, log_file):
        self.cycle_num = cycle_num[0]
        self.log_file = log_file

    def __call__(self, x, df_data, train_frac=TRAIN_FRAC,
                 trainval_frac=TRAINVAL_FRAC, seed=DATA_SPLIT_SEED):
        cycle_ids = df_data[self.cycle_num].unique()
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(cycle_ids)
        train_cycle_ids = cycle_ids[:int(cycle_ids.shape[0] * train_frac)]
        valid_cycle_ids = cycle_ids[int(cycle_ids.shape[0] * train_frac):int(
            cycle_ids.shape[0] * trainval_frac)]
        test_cycle_ids = cycle_ids[int(cycle_ids.shape[0] * trainval_frac):]

        data_indices = np.arange(int(len(df_data)))
        train_slice = data_indices[df_data[self.cycle_num].isin(
            train_cycle_ids)]
        valid_slice = data_indices[df_data[self.cycle_num].isin(
            valid_cycle_ids)]
        test_slice = data_indices[df_data[self.cycle_num].isin(test_cycle_ids)]
        
        print(f'Train {self.cycle_num}: {sorted(train_cycle_ids)}')
        print(f'Valid {self.cycle_num}: {sorted(valid_cycle_ids)}')
        print(f'Test  {self.cycle_num}: {sorted(test_cycle_ids)}')
        
        with open(self.log_file, 'a') as lf:          
            lf.write(f'\nTrain {self.cycle_num}: {sorted(train_cycle_ids)}\n')
            lf.write(f'Valid {self.cycle_num}: {sorted(valid_cycle_ids)}\n')
            lf.write(f'Test  {self.cycle_num}: {sorted(test_cycle_ids)}\n\n')
        lf.close()
        
        return train_slice, valid_slice, test_slice
    
    
class TwoCycleSplitter(Splitter):
    def __init__(self, cycle_nums, log_file):
        self.cycle_nums = cycle_nums
        self.log_file = log_file
        
    def __call__(self, x, df_data, train_frac=TRAIN_FRAC,
                 trainval_frac=TRAINVAL_FRAC, seed=DATA_SPLIT_SEED):
        cycle_a_ids = df_data[self.cycle_nums[0]].unique()
        cycle_b_ids = df_data[self.cycle_nums[1]].unique()
        
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(cycle_a_ids)
        if seed is not None:
            np.random.seed(seed + 1)
        np.random.shuffle(cycle_b_ids)
        
        train_cycle_a_ids = cycle_a_ids[:int(cycle_a_ids.shape[0] * train_frac)]
        train_cycle_b_ids = cycle_b_ids[:int(cycle_b_ids.shape[0] * train_frac)]
        
        valid_cycle_a_ids = cycle_a_ids[int(cycle_a_ids.shape[0] * train_frac):int(
                                cycle_a_ids.shape[0] * trainval_frac)]
        valid_cycle_b_ids = cycle_b_ids[int(cycle_b_ids.shape[0] * train_frac):int(
                                cycle_b_ids.shape[0] * trainval_frac)]
        
        test_cycle_a_ids = cycle_a_ids[int(cycle_a_ids.shape[0] * trainval_frac):]
        test_cycle_b_ids = cycle_b_ids[int(cycle_b_ids.shape[0] * trainval_frac):]
        
        df_data_copy = df_data.copy()
        data_indices = list(np.arange(len(df_data)))
        df_data_copy['idx'] = data_indices
        
        df_test = df_data_copy.loc[
            df_data_copy[self.cycle_nums[0]].isin(test_cycle_a_ids) | 
            df_data_copy[self.cycle_nums[1]].isin(test_cycle_b_ids) 
        ]
        test_slice = list(df_test['idx'])

        df_data_copy = df_data_copy.loc[~df_data_copy['idx'].isin(test_slice)]
        df_valid = df_data_copy.loc[
            df_data_copy[self.cycle_nums[0]].isin(valid_cycle_a_ids) | 
            df_data_copy[self.cycle_nums[1]].isin(valid_cycle_b_ids) 
        ]
        valid_slice = list(df_valid['idx'])

        df_data_copy = df_data_copy.loc[~df_data_copy['idx'].isin(valid_slice)]
        df_train = df_data_copy.loc[
            df_data_copy[self.cycle_nums[0]].isin(train_cycle_a_ids) | 
            df_data_copy[self.cycle_nums[1]].isin(train_cycle_b_ids) 
        ]
        train_slice = list(df_train['idx'])
        
        print(f'Train {self.cycle_nums[0]}: {sorted(train_cycle_a_ids)}')
        print(f'Valid {self.cycle_nums[0]}: {sorted(valid_cycle_a_ids)}')
        print(f'Test  {self.cycle_nums[0]}: {sorted(test_cycle_a_ids)}')
        print()        
        print(f'Train {self.cycle_nums[1]}: {sorted(train_cycle_b_ids)}')
        print(f'Valid {self.cycle_nums[1]}: {sorted(valid_cycle_b_ids)}')
        print(f'Test  {self.cycle_nums[1]}: {sorted(test_cycle_b_ids)}')
        
        with open(self.log_file, 'a') as lf:
            lf.write(f'\nTrain {self.cycle_nums[0]}: {sorted(train_cycle_a_ids)}\n')
            lf.write(f'Valid {self.cycle_nums[0]}: {sorted(valid_cycle_a_ids)}\n')
            lf.write(f'Test  {self.cycle_nums[0]}: {sorted(test_cycle_a_ids)}\n\n')
            
            lf.write(f'Train {self.cycle_nums[1]}: {sorted(train_cycle_b_ids)}\n')
            lf.write(f'Valid {self.cycle_nums[1]}: {sorted(valid_cycle_b_ids)}\n')
            lf.write(f'Test  {self.cycle_nums[1]}: {sorted(test_cycle_b_ids)}\n\n')
        lf.close()
        
        return train_slice, valid_slice, test_slice

    
class ThreeCycleSplitter(Splitter):
    def __init__(self, cycle_nums, log_file):
        self.cycle_nums = cycle_nums
        self.log_file = log_file
        
    def __call__(self, x, df_data, train_frac=TRAIN_FRAC,
                 trainval_frac=TRAINVAL_FRAC, seed=DATA_SPLIT_SEED, 
                 getAllNewTestSlice=False, cyc2_dup_ids=None, cyc3_dup_ids=None):
        cycle_a_ids = df_data[self.cycle_nums[0]].unique()
        cycle_b_ids = df_data[self.cycle_nums[1]].unique()
        cycle_c_ids = df_data[self.cycle_nums[2]].unique()
        
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(cycle_a_ids)
        if seed is not None:
            np.random.seed(seed + 1)
        np.random.shuffle(cycle_b_ids)
        if seed is not None:
            np.random.seed(seed + 2)
        np.random.shuffle(cycle_c_ids)
        
        train_cycle_a_ids = cycle_a_ids[:int(cycle_a_ids.shape[0] * train_frac)]
        train_cycle_b_ids = cycle_b_ids[:int(cycle_b_ids.shape[0] * train_frac)]
        train_cycle_c_ids = cycle_c_ids[:int(cycle_c_ids.shape[0] * train_frac)]

        valid_cycle_a_ids = cycle_a_ids[int(cycle_a_ids.shape[0] * train_frac):int(
                                cycle_a_ids.shape[0] * trainval_frac)]
        valid_cycle_b_ids = cycle_b_ids[int(cycle_b_ids.shape[0] * train_frac):int(
                                cycle_b_ids.shape[0] * trainval_frac)]
        valid_cycle_c_ids = cycle_c_ids[int(cycle_c_ids.shape[0] * train_frac):int(
                                cycle_c_ids.shape[0] * trainval_frac)]

        test_cycle_a_ids = cycle_a_ids[int(cycle_a_ids.shape[0] * trainval_frac):]
        test_cycle_b_ids = cycle_b_ids[int(cycle_b_ids.shape[0] * trainval_frac):]
        test_cycle_c_ids = cycle_c_ids[int(cycle_c_ids.shape[0] * trainval_frac):]

        df_data_copy = df_data.copy()
        data_indices = list(np.arange(len(df_data)))
        df_data_copy['idx'] = data_indices
        
        if not getAllNewTestSlice:
            df_test = df_data_copy.loc[
                df_data_copy[self.cycle_nums[0]].isin(test_cycle_a_ids) | 
                df_data_copy[self.cycle_nums[1]].isin(test_cycle_b_ids) |
                df_data_copy[self.cycle_nums[2]].isin(test_cycle_c_ids)
            ]
            test_slice = list(df_test['idx'])

            df_data_copy = df_data_copy.loc[~df_data_copy['idx'].isin(test_slice)]
            df_valid = df_data_copy.loc[
                df_data_copy[self.cycle_nums[0]].isin(valid_cycle_a_ids) | 
                df_data_copy[self.cycle_nums[1]].isin(valid_cycle_b_ids) |
                df_data_copy[self.cycle_nums[2]].isin(valid_cycle_c_ids)
            ]
            valid_slice = list(df_valid['idx'])

            df_data_copy = df_data_copy.loc[~df_data_copy['idx'].isin(valid_slice)]
            df_train = df_data_copy.loc[
                df_data_copy[self.cycle_nums[0]].isin(train_cycle_a_ids) | 
                df_data_copy[self.cycle_nums[1]].isin(train_cycle_b_ids) |
                df_data_copy[self.cycle_nums[2]].isin(train_cycle_c_ids)
            ]
            train_slice = list(df_train['idx'])

            print(f'Train {self.cycle_nums[0]}: {sorted(train_cycle_a_ids)}')
            print(f'Valid {self.cycle_nums[0]}: {sorted(valid_cycle_a_ids)}')
            print(f'Test  {self.cycle_nums[0]}: {sorted(test_cycle_a_ids)}')
            print()        
            print(f'Train {self.cycle_nums[1]}: {sorted(train_cycle_b_ids)}')
            print(f'Valid {self.cycle_nums[1]}: {sorted(valid_cycle_b_ids)}')
            print(f'Test  {self.cycle_nums[1]}: {sorted(test_cycle_b_ids)}')
            print()
            print(f'Train {self.cycle_nums[2]}: {sorted(train_cycle_c_ids)}')
            print(f'Valid {self.cycle_nums[2]}: {sorted(valid_cycle_c_ids)}')
            print(f'Test  {self.cycle_nums[2]}: {sorted(test_cycle_c_ids)}')

            with open(self.log_file, 'a') as lf:
                lf.write(f'\nTrain {self.cycle_nums[0]}: {sorted(train_cycle_a_ids)}\n')
                lf.write(f'Valid {self.cycle_nums[0]}: {sorted(valid_cycle_a_ids)}\n')
                lf.write(f'Test  {self.cycle_nums[0]}: {sorted(test_cycle_a_ids)}\n\n')

                lf.write(f'Train {self.cycle_nums[1]}: {sorted(train_cycle_b_ids)}\n')
                lf.write(f'Valid {self.cycle_nums[1]}: {sorted(valid_cycle_b_ids)}\n')
                lf.write(f'Test  {self.cycle_nums[1]}: {sorted(test_cycle_b_ids)}\n\n')

                lf.write(f'Train {self.cycle_nums[2]}: {sorted(train_cycle_c_ids)}\n')
                lf.write(f'Valid {self.cycle_nums[2]}: {sorted(valid_cycle_c_ids)}\n')
                lf.write(f'Test  {self.cycle_nums[2]}: {sorted(test_cycle_c_ids)}\n\n')
            lf.close()
        
            return train_slice, valid_slice, test_slice
        
        else:
            indices_to_remove = []
            for i, cyc_c_id in enumerate(test_cycle_c_ids):
                for j, dup_id in enumerate(cyc3_dup_ids):
                    if cyc_c_id == dup_id and cyc2_dup_ids[j] not in test_cycle_b_ids:
                        indices_to_remove.append(i)
                        break
            test_cycle_c_ids = np.delete(test_cycle_c_ids, indices_to_remove)
                        
            indices_to_remove = []
            for i, cyc_b_id in enumerate(test_cycle_b_ids):
                for j, dup_id in enumerate(cyc2_dup_ids):
                    if cyc_b_id == dup_id and cyc3_dup_ids[j] not in test_cycle_c_ids:
                        indices_to_remove.append(i)
                        break
            test_cycle_b_ids = np.delete(test_cycle_b_ids, indices_to_remove)
            
            df_new_test = df_data_copy.loc[
                df_data_copy[self.cycle_nums[0]].isin(test_cycle_a_ids) & 
                df_data_copy[self.cycle_nums[1]].isin(test_cycle_b_ids) &
                df_data_copy[self.cycle_nums[2]].isin(test_cycle_c_ids)
            ]
            
            new_test_slice = list(df_new_test['idx'])
            return new_test_slice
