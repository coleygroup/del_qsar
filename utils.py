import numpy as np 
from tqdm import tqdm

def batch(indices, batch_size, pad=False, sortfunc=lambda x:x):
    batch_indices = np.zeros((batch_size,), dtype=np.int)
    ctr = 0
    for index in tqdm(indices, desc='batching data'):
        batch_indices[ctr] = index
        ctr += 1
        if ctr == batch_size:
            yield sortfunc(batch_indices)
            ctr = 0
    if ctr != 0: # incomplete batch
        if not pad: # return partial batch
            yield sortfunc(batch_indices[:ctr])
            return
        # fill the rest for stability
        while True:
            for index in indices:
                batch_indices[ctr] = index
                ctr += 1
                if ctr == batch_size:
                    yield sortfunc(batch_indices)
                    return
                    