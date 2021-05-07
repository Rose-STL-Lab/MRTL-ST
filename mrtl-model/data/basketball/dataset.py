import logging

import numpy as np
import torch

from config import config

logger = logging.getLogger(config.parent_logger_name).getChild(__name__)


class BballRawDataset(torch.utils.data.Dataset):
    def __init__(self, fn):
        """
        Initialize basketball shot dataset from raw data

        :param fn: filename of data file
        :type fn: str
        """

        # Load data
        # cols = ['team_no', 'possession_no', 'teamID_A', 'teamID_B', 'timestamps', 'quarter', 'a', 'bh_ID', 'bh_x', 'bh_y',
        #         'def1_x', 'def1_y', 'def2_x', 'def2_y', 'def3_x', 'def3_y', 'def4_x', 'def4_y', 'def5_x', 'def5_y',
        #         'def1_ID', 'def2_ID', 'def3_ID', 'def4_ID', 'def5_ID', 'shoot_label']
        self.data = np.load(fn, allow_pickle=True)

        # Set dimensions
        self.a_dims = len(np.unique(self.data.loc[:, 'a']))
        self.b_dims = None
        self.c_dims = None
        # T
        self.t_dims = None
        # T'

        self.a = self.data.loc[:, 'a'].astype(np.int16).to_numpy()
        self.bh_pos = None
        self.def_pos = None
        # T
        self.time = None
        # T'
        self.y = self.data.loc[:, 'shoot_label'].astype(np.uint8).to_numpy()

    def calculate_pos(self, b_dims, c_dims):
        self.b_dims = b_dims
        self.c_dims = c_dims

        scale_bh = config.b_dims[-1][0] / self.b_dims[0]
        scale_def = config.c_dims[-1][0] / self.c_dims[0]

        # Scale bh_pos
        self.bh_pos = (self.data.loc[:, 'bh_x':'bh_y'] / scale_bh).astype(
            np.uint8).to_numpy()

        # Scale def_pos
        invalid_def_pos_val = -100
        def_pos_x = self.data.filter(like='trunc_x')[self.data.filter(
            like='trunc_x') != invalid_def_pos_val]
        def_pos_x = ((def_pos_x + 6) / scale_def)
        def_pos_x = def_pos_x.fillna(c_dims[0]).astype(np.int16).to_numpy()

        def_pos_y = self.data.filter(like='trunc_y')[self.data.filter(
            like='trunc_y') != invalid_def_pos_val]
        def_pos_y = ((def_pos_y + 2) / scale_def)
        def_pos_y = def_pos_y.fillna(c_dims[1]).astype(np.int16).to_numpy()

        # Convert 2D to 1D
        def_pos = def_pos_x * (self.c_dims[0] + 1) + def_pos_y
        mask = torch.zeros(self.data.shape[0],
                           (self.c_dims[0] + 1) * (self.c_dims[1] + 1),
                           dtype=int)
        
        print('mask', mask.shape)
        print('1st param.shape', torch.from_numpy(def_pos).long().shape)
        print('1st param', torch.from_numpy(def_pos).long())
        print('2nd param.shape') 
        print(torch.ones_like(mask).shape)
        print('idk how we r still fine')
        
        mask.scatter_add_(1, 
                          torch.from_numpy(def_pos).long(), 
                          torch.ones_like(mask))
        mask = mask.view(-1, self.c_dims[0] + 1, self.c_dims[1] +
                         1)[:, :self.c_dims[0], :self.c_dims[1]]

        self.def_pos = mask.numpy().astype(np.uint8)
        
    # T
    def calculate_time(self, t_dims):
        self.t_dims = t_dims
        scale_time = config.t_dims[-1] / self.t_dims
        # Scale time
        self.time = (self.data.loc[:, 'quarter'] / scale_time).astype(np.uint8).to_numpy()
    # T'

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.a[idx], self.time[idx], self.bh_pos[
            idx, :], self.def_pos[idx, :, :], self.y[idx]
