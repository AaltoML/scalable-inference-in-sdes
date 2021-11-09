"""Class for the MOCAP data.

Uses the preprocessed data from ODE^2VAE, instructions on data
can be found in https://github.com/cagatayyildiz/ODE2VAE,
and the data download is in the Dropbox folder linked in the README
of the ODE^2VAE repo."""

import tensorflow as tf
import os
from scipy.io import loadmat
import numpy as np




class CMUWalkingTF:
    """A dataset class for the CMU walking dataset.
    """

    def __init__(self, base_folder, data_section='train'):
        """Initialize dataset class.

        Inputs
        ------
            base_folder: str,
                Path to the data folder. Should contain a file with
                name mocap35.mat
            data_section: str,
                Specifies if the data to be retrieved is for training, validation
                or testing.
        """
        self.base_folder = base_folder
        self.data_section = data_section

        data_file = os.path.join(base_folder, 'mocap35.mat') #43 for multiple, 35 for sequences of 1 person
        print(data_file)
        self.data = loadmat(data_file)
        if data_section == 'train':
            self.data = self.data['Xtr']
        elif data_section == 'val':
            self.data = self.data['Xval']
        else:
            self.data = self.data['Xtest']
        print(f'Dataset has {self.data.shape[0]} sequences')

        self.y = tf.convert_to_tensor(self.data, dtype=tf.float64)
        max_len = self.y.shape[1]
        self.x = tf.convert_to_tensor(np.reshape(0.1*np.arange(0, max_len), (max_len, 1)), dtype=tf.float64)
        self.x = tf.stack([self.x]*self.y.shape[0], axis=0)
    
