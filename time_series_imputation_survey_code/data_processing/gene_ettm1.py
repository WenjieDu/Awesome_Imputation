"""
Generate a fully-prepared ETTm1 dataset and save into files for PyPOTS to use.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import pandas as pd
import tsdb
from pygrinder import mcar
from pypots.data.saving import save_dict_into_h5, pickle_dump
from pypots.data.utils import sliding_window
from pypots.utils.logging import logger
from sklearn.preprocessing import StandardScaler

from dataset_config import ARTIFICIALLY_MISSING_RATE

saving_dir = "data/ettm1"
seq_len = 96

if __name__ == "__main__":
    data = tsdb.load("electricity_transformer_temperature")  # load all 4 sub datasets
    df = data["ETTm1"]  # we only need ETTm1
    feature_names = df.columns.tolist()
    feature_num = len(feature_names)
    df["datetime"] = pd.to_datetime(df.index)

    unique_months = df["datetime"].dt.to_period("M").unique()
    selected_as_test = unique_months[:4]  # select first 4 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")
    selected_as_val = unique_months[4:8]  # select the 4th - the 8th months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_train = unique_months[8:]  # use left months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    test_set = df[df["datetime"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["datetime"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["datetime"].dt.to_period("M").isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_set_X = sliding_window(train_set_X, seq_len)
    val_set_X = sliding_window(val_set_X, seq_len)
    test_set_X = sliding_window(test_set_X, seq_len)

    # ETTm1 doesn't contain originally missing values
    # hence we add artificially missing values into the train set here
    train_X = mcar(train_set_X, ARTIFICIALLY_MISSING_RATE)
    train_set_dict = {
        "X": train_X,
    }

    val_X_ori = val_set_X  # reserve for evaluation
    val_X = mcar(val_set_X, ARTIFICIALLY_MISSING_RATE)
    val_set_dict = {
        "X": val_X,
        "X_ori": val_X_ori,
    }

    test_X_ori = test_set_X  # reserve for evaluation
    test_X = mcar(test_set_X, ARTIFICIALLY_MISSING_RATE)
    test_set_dict = {
        "X": test_X,
        "X_ori": test_X_ori,
    }

    save_dict_into_h5(train_set_dict, saving_dir, "train.h5")
    save_dict_into_h5(val_set_dict, saving_dir, "val.h5")
    save_dict_into_h5(test_set_dict, saving_dir, "test.h5")
    pickle_dump(scaler, os.path.join(saving_dir, "scaler.pkl"))

    logger.info(f"Total sample number: {len(train_X)+len(val_X)+len(test_X)}")
    logger.info(f"Number of steps: {train_X.shape[1]}")
    logger.info(f"Number of features: {train_X.shape[2]}")

    logger.info("All done.")
