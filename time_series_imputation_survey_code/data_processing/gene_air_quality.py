"""
Generate a fully-prepared AirQuality dataset and save to into files for PyPOTS to use.
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
from pygrinder import calc_missing_rate
from sklearn.preprocessing import StandardScaler

from dataset_config import ARTIFICIALLY_MISSING_RATE

saving_dir = "data/air_quality"
seq_len = 24

if __name__ == "__main__":
    data = tsdb.load("beijing_multisite_air_quality")
    df = data["X"]
    stations = df["station"].unique()

    df_collector = []
    station_name_collector = []

    for station in stations:
        current_df = df[df["station"] == station]
        logger.info(f"Current dataframe shape: {current_df.shape}")

        current_df["date_time"] = pd.to_datetime(
            current_df[["year", "month", "day", "hour"]]
        )
        station_name_collector.append(current_df.loc[0, "station"])
        # remove duplicated date info and wind direction, which is a categorical col
        current_df = current_df.drop(
            ["year", "month", "day", "hour", "wd", "No", "station"], axis=1
        )
        df_collector.append(current_df)

    logger.info(
        f"There are total {len(station_name_collector)} stations, they are {station_name_collector}"
    )
    date_time = df_collector[0]["date_time"]
    df_collector = [i.drop("date_time", axis=1) for i in df_collector]
    df = pd.concat(df_collector, axis=1)
    feature_names = [
        station + "_" + feature
        for station in station_name_collector
        for feature in df_collector[0].columns
    ]
    feature_num = len(feature_names)
    df.columns = feature_names
    logger.info(
        f"Original df missing rate: "
        f"{(df[feature_names].isna().sum().sum() / (df.shape[0] * feature_num)):.3f}"
    )

    df["date_time"] = date_time
    unique_months = df["date_time"].dt.to_period("M").unique()
    selected_as_test = unique_months[:10]  # select first 3 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")
    selected_as_val = unique_months[10:20]  # select the 4th - the 6th months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_train = unique_months[20:]  # use left months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    test_set = df[df["date_time"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["date_time"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["date_time"].dt.to_period("M").isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_set_X = sliding_window(train_set_X, seq_len)
    val_set_X = sliding_window(val_set_X, seq_len)
    test_set_X = sliding_window(test_set_X, seq_len)

    train_X = train_set_X
    train_set_dict = {
        "X": train_X,
    }
    logger.info(f"Train set missing rate: {calc_missing_rate(train_X)}")

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

    logger.info(f"Total sample number: {len(train_X) + len(val_X) + len(test_X)}")
    logger.info(f"Number of steps: {train_X.shape[1]}")
    logger.info(f"Number of features: {train_X.shape[2]}")

    logger.info("All done.")
