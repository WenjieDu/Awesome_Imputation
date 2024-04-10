"""
Imputation models with tuned hyperparameters for PhysioNet-2012 dataset.
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


def train_models_for_physionet2012(
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
    saits_optimizer = Adam(lr=0.00068277455043675505)
    # set up the SAITS model
    saving_path = os.path.join(result_saving_path, "SAITS_physionet2012")
    saits = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=5,
        d_model=256,
        d_ffn=512,
        n_heads=8,
        d_k=32,
        d_v=32,
        dropout=0,
        attn_dropout=0,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=saits_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    saits.fit(train_set=train_set, val_set=val_set)
    saits_test_set_imputation = saits.predict(test_set)["imputation"]
    saits_mae = calc_mae(saits_test_set_imputation, test_X_ori, test_indicating_mask)
    saits_mse = calc_mse(saits_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"SAITS on PhysioNet-2012: MAE={saits_mae:.4f}, MSE={saits_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    saits_train_set_imputation = saits.predict(train_set)["imputation"]
    saits_val_set_imputation = saits.predict(val_set)["imputation"]
    pickle_dump(
        [
            saits_train_set_imputation,
            saits_val_set_imputation,
            saits_test_set_imputation,
        ],
        os.path.join(saving_path, f"imputation.pkl"),
    )

    # set the learning rate for Transformer
    transformer_optimizer = Adam(lr=0.0005456497467589473)
    # set up the Transformer model
    saving_path = os.path.join(result_saving_path, "Transformer_physionet2012")
    transformer = Transformer(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=5,
        n_heads=4,
        d_k=32,
        d_v=128,
        d_model=128,
        d_ffn=1024,
        dropout=0,
        attn_dropout=0.4,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=transformer_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    transformer.fit(train_set=train_set, val_set=val_set)
    transformer_test_set_imputation = transformer.predict(test_set)["imputation"]
    transformer_mae = calc_mae(
        transformer_test_set_imputation, test_X_ori, test_indicating_mask
    )
    transformer_mse = calc_mse(
        transformer_test_set_imputation, test_X_ori, test_indicating_mask
    )
    logger.info(
        f"Transformer on PhysioNet-2012: MAE={transformer_mae:.4f}, MSE={transformer_mse:.4f}"
    )
    # impute the whole physionet2012 dataset for the downstream classification task
    transformer_train_set_imputation = transformer.predict(train_set)["imputation"]
    transformer_val_set_imputation = transformer.predict(val_set)["imputation"]
    pickle_dump(
        [
            transformer_train_set_imputation,
            transformer_val_set_imputation,
            transformer_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for TimesNet
    timesnet_optimizer = Adam(lr=0.00359347659459279)
    # set up the TimesNet model
    saving_path = os.path.join(result_saving_path, "TimesNet_physionet2012")
    timesnet = TimesNet(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=1,
        top_k=1,
        d_model=128,
        d_ffn=512,
        n_kernels=5,
        dropout=0.5,
        apply_nonstationary_norm=True,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=timesnet_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    timesnet.fit(train_set=train_set, val_set=val_set)
    timesnet_test_set_imputation = timesnet.predict(test_set)["imputation"]
    timesnet_mae = calc_mae(
        timesnet_test_set_imputation, test_X_ori, test_indicating_mask
    )
    timesnet_mse = calc_mse(
        timesnet_test_set_imputation, test_X_ori, test_indicating_mask
    )
    logger.info(
        f"TimesNet on PhysioNet-2012: MAE={timesnet_mae:.4f}, MSE={timesnet_mse:.4f}"
    )
    # impute the whole physionet2012 dataset for the downstream classification task
    timesnet_train_set_imputation = timesnet.predict(train_set)["imputation"]
    timesnet_val_set_imputation = timesnet.predict(val_set)["imputation"]
    pickle_dump(
        [
            timesnet_train_set_imputation,
            timesnet_val_set_imputation,
            timesnet_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for CSDI
    csdi_optimizer = Adam(lr=0.0008650784451042799)
    # set up the CSDI model
    saving_path = os.path.join(result_saving_path, "CSDI_physionet2012")
    csdi = CSDI(
        n_features=n_features,
        n_layers=6,
        n_heads=2,
        n_channels=128,
        d_time_embedding=64,
        d_feature_embedding=32,
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
    csdi_test_set_imputation = csdi.predict(test_set, n_sampling_times=10)["imputation"]
    mean_csdi_test_set_imputation = csdi_test_set_imputation.mean(axis=1)
    csdi_mae = calc_mae(mean_csdi_test_set_imputation, test_X_ori, test_indicating_mask)
    csdi_mse = calc_mse(mean_csdi_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"CSDI on PhysioNet-2012: MAE={csdi_mae:.4f}, MSE={csdi_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    csdi_train_set_imputation = csdi.predict(train_set, n_sampling_times=10)[
        "imputation"
    ]
    mean_csdi_train_set_imputation = csdi_train_set_imputation.mean(axis=1)
    csdi_val_set_imputation = csdi.predict(val_set, n_sampling_times=10)["imputation"]
    mean_csdi_val_set_imputation = csdi_val_set_imputation.mean(axis=1)
    pickle_dump(
        [
            mean_csdi_train_set_imputation,
            mean_csdi_val_set_imputation,
            mean_csdi_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for GP-VAE
    gpvae_optimizer = Adam(lr=0.00014953414131156273)
    # set up the GP-VAE model
    saving_path = os.path.join(result_saving_path, "GPVAE_physionet2012")
    gpvae = GPVAE(
        n_steps=n_steps,
        n_features=n_features,
        latent_size=n_features,
        encoder_sizes=(512, 512),
        decoder_sizes=(128, 128),
        beta=0.2,
        length_scale=7,
        sigma=1.005,
        window_size=36,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=gpvae_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    gpvae.fit(train_set=train_set, val_set=val_set)
    gpvae_test_set_imputation = gpvae.predict(test_set)["imputation"]
    mean_gpvae_test_set_imputation = gpvae_test_set_imputation.mean(axis=1)
    gpvae_mae = calc_mae(
        mean_gpvae_test_set_imputation, test_X_ori, test_indicating_mask
    )
    gpvae_mse = calc_mse(
        mean_gpvae_test_set_imputation, test_X_ori, test_indicating_mask
    )
    logger.info(f"GP-VAE on PhysioNet-2012: MAE={gpvae_mae:.4f}, MSE={gpvae_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    gpvae_train_set_imputation = gpvae.predict(train_set)["imputation"]
    mean_gpvae_train_set_imputation = gpvae_train_set_imputation.mean(axis=1)
    gpvae_val_set_imputation = gpvae.predict(val_set)["imputation"]
    mean_gpvae_val_set_imputation = gpvae_val_set_imputation.mean(axis=1)
    pickle_dump(
        [
            mean_gpvae_train_set_imputation,
            mean_gpvae_val_set_imputation,
            mean_gpvae_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for US-GAN generator and discriminator
    usgan_G_optimizer = Adam(lr=0.000156727827009108)
    usgan_D_optimizer = Adam(lr=0.000156727827009108)
    # set up the US-GAN model
    saving_path = os.path.join(result_saving_path, "USGAN_physionet2012")
    usgan = USGAN(
        n_steps=n_steps,
        n_features=n_features,
        rnn_hidden_size=1024,
        dropout=0.4,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        G_optimizer=usgan_G_optimizer,
        D_optimizer=usgan_D_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    usgan.fit(train_set=train_set, val_set=val_set)
    usgan_test_set_imputation = usgan.predict(test_set)["imputation"]
    usgan_mae = calc_mae(usgan_test_set_imputation, test_X_ori, test_indicating_mask)
    usgan_mse = calc_mse(usgan_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"US-GAN on PhysioNet-2012: MAE={usgan_mae:.4f}, MSE={usgan_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    usgan_train_set_imputation = usgan.predict(train_set)["imputation"]
    usgan_val_set_imputation = usgan.predict(val_set)["imputation"]
    pickle_dump(
        [
            usgan_train_set_imputation,
            usgan_val_set_imputation,
            usgan_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for BRITS
    brits_optimizer = Adam(lr=0.0005211846755855906)
    # set up the BRITS model
    saving_path = os.path.join(result_saving_path, "BRITS_physionet2012")
    brits = BRITS(
        n_steps=n_steps,
        n_features=n_features,
        rnn_hidden_size=1024,
        batch_size=BATCH_SIZE,
        epochs=MAX_N_EPOCHS,
        patience=PATIENCE,
        optimizer=brits_optimizer,
        device=DEVICE,
        saving_path=saving_path,
    )
    brits.fit(train_set=train_set, val_set=val_set)
    brits_test_set_imputation = brits.predict(test_set)["imputation"]
    brits_mae = calc_mae(brits_test_set_imputation, test_X_ori, test_indicating_mask)
    brits_mse = calc_mse(brits_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"BRITS on PhysioNet-2012: MAE={brits_mae:.4f}, MSE={brits_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    brits_train_set_imputation = brits.predict(train_set)["imputation"]
    brits_val_set_imputation = brits.predict(val_set)["imputation"]
    pickle_dump(
        [
            brits_train_set_imputation,
            brits_val_set_imputation,
            brits_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set the learning rate for M-RNN
    mrnn_optimizer = Adam(lr=0.0015892258071479781)
    # set up the MRNN model
    saving_path = os.path.join(result_saving_path, "MRNN_physionet2012")
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
    mrnn_test_set_imputation = mrnn.predict(test_set)["imputation"]
    mrnn_mae = calc_mae(mrnn_test_set_imputation, test_X_ori, test_indicating_mask)
    mrnn_mse = calc_mse(mrnn_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"MRNN on PhysioNet-2012: MAE={mrnn_mae:.4f}, MSE={mrnn_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    mrnn_train_set_imputation = mrnn.predict(train_set)["imputation"]
    mrnn_val_set_imputation = mrnn.predict(val_set)["imputation"]
    pickle_dump(
        [
            mrnn_train_set_imputation,
            mrnn_val_set_imputation,
            mrnn_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set up the LOCF method
    saving_path = os.path.join(result_saving_path, "LOCF_physionet2012")
    locf = LOCF(device="cpu")
    locf_test_set_imputation = locf.predict(test_set)["imputation"]
    locf_mae = calc_mae(locf_test_set_imputation, test_X_ori, test_indicating_mask)
    locf_mse = calc_mse(locf_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"LOCF on PhysioNet-2012: MAE={locf_mae:.4f}, MSE={locf_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    locf_train_set_imputation = locf.predict(train_set)["imputation"]
    locf_val_set_imputation = locf.predict(val_set)["imputation"]
    pickle_dump(
        [
            locf_train_set_imputation,
            locf_val_set_imputation,
            locf_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set up the median imputation method
    saving_path = os.path.join(result_saving_path, "Median_physionet2012")
    median_test_set_imputation = median_imputation(test_X)
    median_mae = calc_mae(median_test_set_imputation, test_X_ori, test_indicating_mask)
    median_mse = calc_mse(median_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"Median on PhysioNet-2012: MAE={median_mae:.4f}, MSE={median_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    median_train_set_imputation = median_imputation(train_set)
    median_val_set_imputation = median_imputation(val_set)
    pickle_dump(
        [
            median_train_set_imputation,
            median_val_set_imputation,
            median_test_set_imputation,
        ],
        os.path.join(saving_path, "imputation.pkl"),
    )

    # set up the mean imputation method
    saving_path = os.path.join(result_saving_path, "Mean_physionet2012")
    mean_test_set_imputation = mean_imputation(test_X)
    mean_mae = calc_mae(mean_test_set_imputation, test_X_ori, test_indicating_mask)
    mean_mse = calc_mse(mean_test_set_imputation, test_X_ori, test_indicating_mask)
    logger.info(f"Mean on PhysioNet-2012: MAE={mean_mae:.4f}, MSE={mean_mse:.4f}")
    # impute the whole physionet2012 dataset for the downstream classification task
    mean_train_set_imputation = mean_imputation(train_set)
    mean_val_set_imputation = mean_imputation(val_set)
    pickle_dump(
        [
            mean_train_set_imputation,
            mean_val_set_imputation,
            mean_test_set_imputation,
        ],
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
    dataset = "data/physionet_2012"
    train_and_sum(dataset, train_models_for_physionet2012)
