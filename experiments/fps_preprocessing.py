import os
import sys
import pandas as pd
import h5py

from absl import app
from absl import flags

import rdkit
print(rdkit.__version__)

DELQSAR_ROOT = os.path.abspath(__file__ + '/../../')
sys.path += [os.path.dirname(DELQSAR_ROOT)]

from del_qsar import featurizers

sys.path += ['experiments']

FLAGS = flags.FLAGS
flags.DEFINE_string('csv', 'DD1S_CAIX_QSAR.csv', 'csv filename')
flags.DEFINE_string('fps_h5', 'x_DD1S_CAIX_2048_bits_all_fps.h5', 'filename for HDF5 file to create')
flags.DEFINE_integer('fp_size', 2048, 'fingerprint length (number of bits)')

def main(argv):
    del argv
    
    # Get data
    df_data = pd.read_csv(os.path.join(DELQSAR_ROOT, 'experiments', 'datasets', FLAGS.csv))

    # Featurizer
    featurizer = featurizers.FingerprintFeaturizer(fp_size=FLAGS.fp_size)
    x = featurizer.prepare_x(df_data)
    try:
        hf = h5py.File(FLAGS.fps_h5, 'w')
        hf.create_dataset('all_fps', data=x)
        hf.close()
        print(f'All fingerprints generated and stored in {FLAGS.fps_h5}')
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    app.run(main)
    