"""
Utility functions.
"""

import os

import h5py
import numpy as np

from global_config import LAZY_LOAD_DATA


def get_datasets_path(data_dir):
    train_set_path = os.path.join(data_dir, "train.h5")
    val_set_path = os.path.join(data_dir, "val.h5")
    test_set_path = os.path.join(data_dir, "test.h5")

    if LAZY_LOAD_DATA:
        # if LAZY_LOAD, only need to provide the dataset file path to PyPOTS models
        prepared_train_set = train_set_path
        prepared_val_set = val_set_path
    else:
        # if not LAZY_LOAD, extract and organize the data into dictionaries for PyPOTS models
        with h5py.File(train_set_path, "r") as hf:
            train_X_arr = hf["X"][:]
        with h5py.File(val_set_path, "r") as hf:
            val_X_arr = hf["X"][:]
            val_X_ori_arr = hf["X_ori"][:]

        prepared_train_set = {"X": train_X_arr}
        prepared_val_set = {"X": val_X_arr, "X_ori": val_X_ori_arr}

    with h5py.File(test_set_path, "r") as hf:
        test_X_arr = hf["X"][:]
        test_X_ori_arr = hf["X_ori"][:]  # need test_X_ori_arr to calculate MAE and MSE

    test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    test_X_ori_arr = np.nan_to_num(test_X_ori_arr)
    return (
        prepared_train_set,
        prepared_val_set,
        test_X_arr,
        test_X_ori_arr,
        test_indicating_arr,
    )
