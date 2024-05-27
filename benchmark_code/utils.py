"""
Utility functions.
"""

import os

import h5py
import numpy as np
import torch
from pypots.utils.logging import logger
from pypots.utils.random import set_random_seed

from global_config import (
    LAZY_LOAD_DATA,
    TORCH_N_THREADS,
    RESULTS_SAVING_PATH,
    RANDOM_SEEDS,
)


def median_imputation(X):
    if isinstance(X, dict):
        X = X["X"]
    elif isinstance(X, str):
        with h5py.File(X, "r") as hf:
            X = hf["X"][:]
    elif isinstance(X, np.ndarray):
        pass
    else:
        raise ValueError(
            "X should be a dict, a path str to an h5 file, or a numpy array"
        )

    assert len(X.shape) == 3, "X should be a 3D array (n_samples, n_steps, n_features)"
    n_samples, n_steps, n_features = X.shape
    X_imputed_reshaped = np.copy(X).reshape(-1, n_features)
    median_values = np.nanmedian(X_imputed_reshaped, axis=0)
    for i, v in enumerate(median_values):
        X_imputed_reshaped[:, i] = np.nan_to_num(X_imputed_reshaped[:, i], nan=v)
    imputation = X_imputed_reshaped.reshape(-1, n_steps, n_features)
    return imputation


def mean_imputation(X):
    if isinstance(X, dict):
        X = X["X"]
    elif isinstance(X, str):
        with h5py.File(X, "r") as hf:
            X = hf["X"][:]
    elif isinstance(X, np.ndarray):
        pass
    else:
        raise ValueError(
            "X should be a dict, a path str to an h5 file, or a numpy array"
        )

    assert len(X.shape) == 3, "X should be a 3D array (n_samples, n_steps, n_features)"
    n_samples, n_steps, n_features = X.shape
    X_imputed_reshaped = np.copy(X).reshape(-1, n_features)
    mean_values = np.nanmean(X_imputed_reshaped, axis=0)
    for i, v in enumerate(mean_values):
        X_imputed_reshaped[:, i] = np.nan_to_num(X_imputed_reshaped[:, i], nan=v)
    imputation = X_imputed_reshaped.reshape(-1, n_steps, n_features)
    return imputation


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


def train_and_sum(dataset, training_func):
    # set the number of threads for pytorch
    torch.set_num_threads(TORCH_N_THREADS)

    saits_mae_collector = []
    saits_mse_collector = []
    transformer_mae_collector = []
    transformer_mse_collector = []
    timesnet_mae_collector = []
    timesnet_mse_collector = []
    csdi_mae_collector = []
    csdi_mse_collector = []
    gpvae_mae_collector = []
    gpvae_mse_collector = []
    usgan_mae_collector = []
    usgan_mse_collector = []
    brits_mae_collector = []
    brits_mse_collector = []
    mrnn_mae_collector = []
    mrnn_mse_collector = []
    locf_mae_collector = []
    locf_mse_collector = []
    median_mae_collector = []
    median_mse_collector = []
    mean_mae_collector = []
    mean_mse_collector = []

    for n_round in range(5):
        set_random_seed(RANDOM_SEEDS[n_round])
        result_saving_path = os.path.join(RESULTS_SAVING_PATH, f"round_{n_round}")
        (
            train_set,
            val_set,
            test_X,
            test_X_ori,
            test_indicating_mask,
        ) = get_datasets_path(dataset)

        (
            saits_mae,
            saits_mse,
            transformer_mae,
            transformer_mse,
            timesnet_mae,
            timesnet_mse,
            csdi_mae,
            csdi_mse,
            gpvae_mae,
            gpvae_mse,
            usgan_mae,
            usgan_mse,
            brits_mae,
            brits_mse,
            mrnn_mae,
            mrnn_mse,
            locf_mae,
            locf_mse,
            median_mae,
            median_mse,
            mean_mae,
            mean_mse,
        ) = training_func(
            train_set,
            val_set,
            test_X,
            test_X_ori,
            test_indicating_mask,
            result_saving_path,
        )

        saits_mae_collector.append(saits_mae)
        saits_mse_collector.append(saits_mse)
        transformer_mae_collector.append(transformer_mae)
        transformer_mse_collector.append(transformer_mse)
        timesnet_mae_collector.append(timesnet_mae)
        timesnet_mse_collector.append(timesnet_mse)
        csdi_mae_collector.append(csdi_mae)
        csdi_mse_collector.append(csdi_mse)
        gpvae_mae_collector.append(gpvae_mae)
        gpvae_mse_collector.append(gpvae_mse)
        usgan_mae_collector.append(usgan_mae)
        usgan_mse_collector.append(usgan_mse)
        brits_mae_collector.append(brits_mae)
        brits_mse_collector.append(brits_mse)
        mrnn_mae_collector.append(mrnn_mae)
        mrnn_mse_collector.append(mrnn_mse)
        locf_mae_collector.append(locf_mae)
        locf_mse_collector.append(locf_mse)
        median_mae_collector.append(median_mae)
        median_mse_collector.append(median_mse)
        mean_mae_collector.append(mean_mae)
        mean_mse_collector.append(mean_mse)

    logger.info(
        f"SAITS on {dataset}: MAE={np.mean(saits_mae_collector):.3f}±{np.std(saits_mae_collector)}, "
        f"MSE={np.mean(saits_mse_collector):.3f}±{np.std(saits_mse_collector)}\n"
        f"Transformer on {dataset}: MAE={np.mean(transformer_mae_collector):.3f}±{np.std(transformer_mae_collector)}, "
        f"MSE={np.mean(transformer_mse_collector):.3f}±{np.std(transformer_mse_collector)}\n"
        f"TimesNet on {dataset}: MAE={np.mean(timesnet_mae_collector):.3f}±{np.std(timesnet_mae_collector)}, "
        f"MSE={np.mean(timesnet_mse_collector):.3f}±{np.std(timesnet_mse_collector)}\n"
        f"CSDI on {dataset}: MAE={np.mean(csdi_mae_collector):.3f}±{np.std(csdi_mae_collector)}, "
        f"MSE={np.mean(csdi_mse_collector):.3f}±{np.std(csdi_mse_collector)}\n"
        f"GPVAE on {dataset}: MAE={np.mean(gpvae_mae_collector):.3f}±{np.std(gpvae_mae_collector)}, "
        f"MSE={np.mean(gpvae_mse_collector):.3f}±{np.std(gpvae_mse_collector)}\n"
        f"USGAN on {dataset}: MAE={np.mean(usgan_mae_collector):.3f}±{np.std(usgan_mae_collector)}, "
        f"MSE={np.mean(usgan_mse_collector):.3f}±{np.std(usgan_mse_collector)}\n"
        f"BRITS on {dataset}: MAE={np.mean(brits_mae_collector):.3f}±{np.std(brits_mae_collector)}, "
        f"MSE={np.mean(brits_mse_collector):.3f}±{np.std(brits_mse_collector)}\n"
        f"MRNN on {dataset}: MAE={np.mean(mrnn_mae_collector):.3f}±{np.std(mrnn_mae_collector)}, "
        f"MSE={np.mean(mrnn_mse_collector):.3f}±{np.std(mrnn_mse_collector)}\n"
        f"LOCF on {dataset}: MAE={np.mean(locf_mae_collector):.3f}±{np.std(locf_mae_collector)}, "
        f"MSE={np.mean(locf_mse_collector):.3f}±{np.std(locf_mse_collector)}\n"
        f"Median on {dataset}: MAE={np.mean(median_mae_collector):.3f}±{np.std(median_mae_collector)}, "
        f"MSE={np.mean(median_mse_collector):.3f}±{np.std(median_mse_collector)}\n"
        f"Mean on {dataset}: MAE={np.mean(mean_mae_collector):.3f}±{np.std(mean_mae_collector)}, "
        f"MSE={np.mean(mean_mse_collector):.3f}±{np.std(mean_mse_collector)}\n"
    )
