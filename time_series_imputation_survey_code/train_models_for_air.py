"""
Imputation models with tuned hyperparameters for Air Quality dataset.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import numpy as np
from pypots.data.saving import pickle_dump
from pypots.imputation import (
    SAITS,
    Transformer,
    TimesNet,
    CSDI,
    GPVAE,
    USGAN,
    BRITS,
    MRNN,
    LOCF,
)
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse

from global_config import BATCH_SIZE, MAX_N_EPOCHS, PATIENCE, DEVICE
from utils import train_and_sum, median_imputation, mean_imputation


def train_models_for_air_quality(
    train_set: str,
    val_set: str,
    test_X: np.ndarray,
    test_X_ori: np.ndarray,
    test_indicating_mask: np.ndarray,
    result_saving_path: str,
):
    _, n_steps, n_features = test_X.shape
    test_set = {"X": test_X}

    # set the learning rate for SAITS
    saits_optimizer = Adam(lr=0.000060981966280654102)
    # set up the SAITS model
    saving_path = os.path.join(result_saving_path, "SAITS_air_quality")
    saits = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=1,
        d_model=2048,
        d_ffn=1024,
        n_heads=4,
        d_k=512,
        d_v=512,
        dropout=0.1,
        attn_dropout=0.1,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=saits_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    saits.fit(train_set=train_set, val_set=val_set)
    saits_results = saits.predict(test_set)
    saits_imputation = saits_results["imputation"]
    saits_mae = calc_mae(saits_imputation, test_X_ori, test_indicating_mask)
    saits_mse = calc_mse(saits_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"SAITS on Air-Quality: MAE={saits_mae:.4f}, MSE={saits_mse:.4f}")
    pickle_dump(
        saits_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for Transformer
    transformer_optimizer = Adam(lr=0.00010671246633083336)
    # set up the Transformer model
    saving_path = os.path.join(result_saving_path, "Transformer_air_quality")
    transformer = Transformer(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=2,
        n_heads=8,
        d_k=128,
        d_v=128,
        d_model=1024,
        d_ffn=1024,
        dropout=0,
        attn_dropout=0.1,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=transformer_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    transformer.fit(train_set=train_set, val_set=val_set)
    transformer_results = transformer.predict(test_set)
    transformer_imputation = transformer_results["imputation"]
    transformer_mae = calc_mae(transformer_imputation, test_X_ori, test_indicating_mask)
    transformer_mse = calc_mse(transformer_imputation, test_X_ori, test_indicating_mask)
    logger.info(
        f"Transformer on Air-Quality: MAE={transformer_mae:.4f}, MSE={transformer_mse:.4f}"
    )
    pickle_dump(
        transformer_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for TimesNet
    timesnet_optimizer = Adam(lr=0.00021376744551644943)
    # set up the TimesNet model
    saving_path = os.path.join(result_saving_path, "TimesNet_air_quality")
    timesnet = TimesNet(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=1,
        top_k=2,
        d_model=512,
        d_ffn=512,
        n_kernels=4,
        dropout=0.2,
        apply_nonstationary_norm=True,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=timesnet_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    timesnet.fit(train_set=train_set, val_set=val_set)
    timesnet_results = timesnet.predict(test_set)
    timesnet_imputation = timesnet_results["imputation"]
    timesnet_mae = calc_mae(timesnet_imputation, test_X_ori, test_indicating_mask)
    timesnet_mse = calc_mse(timesnet_imputation, test_X_ori, test_indicating_mask)
    logger.info(
        f"TimesNet on Air-Quality: MAE={timesnet_mae:.4f}, MSE={timesnet_mse:.4f}"
    )
    pickle_dump(
        timesnet_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for CSDI
    csdi_optimizer = Adam(lr=0.0032519038841160855)
    # set up the CSDI model
    saving_path = os.path.join(result_saving_path, "CSDI_air_quality")
    csdi = CSDI(
        n_features=n_features,
        n_layers=6,
        n_heads=2,
        n_channels=32,
        d_time_embedding=256,
        d_feature_embedding=8,
        d_diffusion_embedding=128,
        target_strategy="random",
        n_diffusion_steps=50,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=csdi_optimizer,
        device=DEVICE,
        saving_path=saving_path,
        model_saving_strategy="all",
    )
    csdi.fit(train_set=train_set, val_set=val_set)
    csdi_results = csdi.predict(test_set, n_sampling_times=10)
    csdi_imputation = csdi_results["imputation"]
    mean_csdi_imputation = csdi_imputation.mean(axis=1)
    csdi_mae = calc_mae(mean_csdi_imputation, test_X_ori, test_indicating_mask)
    csdi_mse = calc_mse(mean_csdi_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"CSDI on Air-Quality: MAE={csdi_mae:.4f}, MSE={csdi_mse:.4f}")
    pickle_dump(
        csdi_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for GP-VAE
    gpvae_optimizer = Adam(lr=0.0002613681169717996)
    # set up the GP-VAE model
    saving_path = os.path.join(result_saving_path, "GPVAE_air_quality")
    gpvae = GPVAE(
        n_steps=n_steps,
        n_features=n_features,
        latent_size=n_features,
        encoder_sizes=(512, 512),
        decoder_sizes=(512, 512),
        beta=0.2,
        length_scale=7,
        sigma=1.005,
        window_size=24,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=gpvae_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    gpvae.fit(train_set=train_set, val_set=val_set)
    gpvae_results = gpvae.predict(test_set)
    gpvae_imputation = gpvae_results["imputation"]
    mean_gpvae_imputation = gpvae_imputation.mean(axis=1)
    gpvae_mae = calc_mae(mean_gpvae_imputation, test_X_ori, test_indicating_mask)
    gpvae_mse = calc_mse(mean_gpvae_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"GP-VAE on Air-Quality: MAE={gpvae_mae:.4f}, MSE={gpvae_mse:.4f}")
    pickle_dump(
        gpvae_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for US-GAN generator and discriminator
    usgan_G_optimizer = Adam(lr=0.000665752360070025)
    usgan_D_optimizer = Adam(lr=0.000665752360070025)
    # set up the US-GAN model
    saving_path = os.path.join(result_saving_path, "USGAN_air_quality")
    usgan = USGAN(
        n_steps=n_steps,
        n_features=n_features,
        rnn_hidden_size=128,
        dropout=0.5,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        G_optimizer=usgan_G_optimizer,
        D_optimizer=usgan_D_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    usgan.fit(train_set=train_set, val_set=val_set)
    usgan_results = usgan.predict(test_set)
    usgan_imputation = usgan_results["imputation"]
    usgan_mae = calc_mae(usgan_imputation, test_X_ori, test_indicating_mask)
    usgan_mse = calc_mse(usgan_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"US-GAN on Air-Quality: MAE={usgan_mae:.4f}, MSE={usgan_mse:.4f}")
    pickle_dump(
        usgan_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for BRITS
    brits_optimizer = Adam(lr=0.0001277432738192592)
    # set up the BRITS model
    saving_path = os.path.join(result_saving_path, "BRITS_air_quality")
    brits = BRITS(
        n_steps=n_steps,
        n_features=n_features,
        rnn_hidden_size=256,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=brits_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    brits.fit(train_set=train_set, val_set=val_set)
    brits_results = brits.predict(test_set)
    brits_imputation = brits_results["imputation"]
    brits_mae = calc_mae(brits_imputation, test_X_ori, test_indicating_mask)
    brits_mse = calc_mse(brits_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"BRITS on Air-Quality: MAE={brits_mae:.4f}, MSE={brits_mse:.4f}")
    pickle_dump(
        brits_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for M-RNN
    mrnn_optimizer = Adam(lr=0.0059824462759454095)
    # set up the MRNN model
    saving_path = os.path.join(result_saving_path, "MRNN_air_quality")
    mrnn = MRNN(
        n_steps=n_steps,
        n_features=n_features,
        rnn_hidden_size=16,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=mrnn_optimizer,
        device=DEVICE,
        saving_path=saving_path,
        model_saving_strategy="all",
    )
    mrnn.fit(train_set=train_set, val_set=val_set)
    mrnn_results = mrnn.predict(test_set)
    mrnn_imputation = mrnn_results["imputation"]
    mrnn_mae = calc_mae(mrnn_imputation, test_X_ori, test_indicating_mask)
    mrnn_mse = calc_mse(mrnn_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"MRNN on Air-Quality: MAE={mrnn_mae:.4f}, MSE={mrnn_mse:.4f}")
    pickle_dump(
        mrnn_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set up the LOCF method
    saving_path = os.path.join(result_saving_path, "LOCF_air_quality")
    locf = LOCF(device="cpu")
    locf_results = locf.predict(test_set)
    locf_imputation = locf_results["imputation"]
    locf_mae = calc_mae(locf_imputation, test_X_ori, test_indicating_mask)
    locf_mse = calc_mse(locf_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"LOCF on Air-Quality: MAE={locf_mae:.4f}, MSE={locf_mse:.4f}")
    pickle_dump(
        locf_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set up the median imputation method
    saving_path = os.path.join(result_saving_path, "Median_air_quality")
    median_test_set_imputation = median_imputation(test_X)
    median_mae = calc_mae(median_test_set_imputation, test_X_ori, test_indicating_mask)
    median_mse = calc_mse(median_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"Median on Air-Quality: MAE={median_mae:.4f}, MSE={median_mse:.4f}")
    pickle_dump(
        median_test_set_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set up the mean imputation method
    saving_path = os.path.join(result_saving_path, "Mean_air_quality")
    mean_test_set_imputation = mean_imputation(test_X)
    mean_mae = calc_mae(mean_test_set_imputation, test_X_ori, test_indicating_mask)
    mean_mse = calc_mse(mean_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"Mean on Air-Quality: MAE={mean_mae:.4f}, MSE={mean_mse:.4f}")
    pickle_dump(
        mean_test_set_imputation,
        os.path.join(saving_path, "imputation.pkl"),
    )

    return (
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
    )


if __name__ == "__main__":
    dataset = "data/air_quality"
    train_and_sum(dataset, train_models_for_air_quality)
