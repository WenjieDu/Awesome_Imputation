"""
Generate a fully-prepared PhysioNet-2012 dataset and save to into files for PyPOTS to use.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import pandas as pd
import tsdb
from pygrinder import mcar
from pypots.data.saving import save_dict_into_h5, pickle_dump
from pypots.utils.logging import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_config import ARTIFICIALLY_MISSING_RATE

saving_dir = "data/physionet_2012"

if __name__ == "__main__":
    data = tsdb.load("physionet_2012")
    X = data["X"].drop(data["static_features"], axis=1)
    y = data["y"]

    def apply_func(df_temp):  # pad and truncate to set the max length of samples as 48
        missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))
        missing_part = pd.DataFrame({"Time": missing})
        df_temp = pd.concat(
            [df_temp, missing_part], ignore_index=False, sort=False
        )  # pad the sample's length to 48 if it doesn't have enough time steps
        df_temp = df_temp.set_index("Time").sort_index().reset_index()
        df_temp = df_temp.iloc[:48]  # truncate
        return df_temp

    X = X.groupby("RecordID").apply(apply_func)
    X = X.drop("RecordID", axis=1)
    X = X.reset_index()
    X = X.drop(["level_1"], axis=1)

    # generate samples
    all_recordID = X["RecordID"].unique()
    train_set_ids, test_set_ids = train_test_split(all_recordID, test_size=0.2)
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=0.2)
    train_set_ids.sort()
    val_set_ids.sort()
    test_set_ids.sort()
    train_set = X[X["RecordID"].isin(train_set_ids)].sort_values(["RecordID", "Time"])
    val_set = X[X["RecordID"].isin(val_set_ids)].sort_values(["RecordID", "Time"])
    test_set = X[X["RecordID"].isin(test_set_ids)].sort_values(["RecordID", "Time"])

    train_set = train_set.drop(["RecordID", "Time"], axis=1)
    val_set = val_set.drop(["RecordID", "Time"], axis=1)
    test_set = test_set.drop(["RecordID", "Time"], axis=1)
    train_X, val_X, test_X = (
        train_set.to_numpy(),
        val_set.to_numpy(),
        test_set.to_numpy(),
    )

    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    # reshape into time series samples
    train_X = train_X.reshape(len(train_set_ids), 48, -1)
    val_X = val_X.reshape(len(val_set_ids), 48, -1)
    test_X = test_X.reshape(len(test_set_ids), 48, -1)

    train_y = y[y.index.isin(train_set_ids)].sort_index()
    val_y = y[y.index.isin(val_set_ids)].sort_index()
    test_y = y[y.index.isin(test_set_ids)].sort_index()
    train_y, val_y, test_y = train_y.to_numpy(), val_y.to_numpy(), test_y.to_numpy()

    train_set_dict = {
        "X": train_X,
        "y": train_y.flatten(),
    }

    # mask values in the validation set as ground truth
    val_X_ori = val_X
    val_X = mcar(val_X, ARTIFICIALLY_MISSING_RATE)
    val_set_dict = {
        "X": val_X,
        "X_ori": val_X_ori,
        "y": val_y.flatten(),
    }

    # mask values in the test set as ground truth
    test_X_ori = test_X
    test_X = mcar(test_X, ARTIFICIALLY_MISSING_RATE)
    test_set_dict = {
        "X": test_X,
        "X_ori": test_X_ori,
        "y": test_y.flatten(),
    }

    save_dict_into_h5(train_set_dict, saving_dir, "train.h5")
    save_dict_into_h5(val_set_dict, saving_dir, "val.h5")
    save_dict_into_h5(test_set_dict, saving_dir, "test.h5")
    pickle_dump(scaler, os.path.join(saving_dir, "scaler.pkl"))

    logger.info(f"Total sample number: {len(train_X) + len(val_X) + len(test_X)}")
    logger.info(f"Number of steps: {train_X.shape[1]}")
    logger.info(f"Number of features: {train_X.shape[2]}")

    logger.info("All done.")
