"""
A simple RNN classifier for downstream classification task on imputed Physionet 2012 dataset.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from pypots.data.saving import load_dict_from_h5
from pypots.nn.modules.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
)
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBRegressor

from global_config import RANDOM_SEEDS


class LoadImputedData(Dataset):
    def __init__(self, imputed_input, target):
        self.imputed_input = imputed_input
        self.target = target

    def __len__(self):
        return len(self.imputed_input)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.imputed_input[idx]).to(torch.float32),
            torch.from_numpy(self.target[idx]).to(torch.float32),
        )


class SimpleRNNForecaster(torch.nn.Module):
    def __init__(
        self, n_features, n_steps, rnn_hidden_size, n_out_features, n_out_steps
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_out_steps = n_out_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = torch.nn.LSTM(
            n_features,
            hidden_size=rnn_hidden_size,
            batch_first=True,
        )
        self.fcn_out = torch.nn.Linear(rnn_hidden_size, n_out_features)

    def forward(self, data):
        estimations = []
        for i in range(self.n_out_steps):
            X = data[:, i : i + self.n_steps]
            hidden_states, _ = self.rnn(X)
            estimation = self.fcn_out(hidden_states[:, -1])
            estimations.append(estimation)

        output = torch.stack(estimations, dim=1)
        return output


class TransformerForecaster(torch.nn.Module):
    def __init__(
        self,
        n_features,
        n_steps,
        n_out_features,
        n_out_steps,
        n_layers,
        d_model,
        n_heads,
        d_ffn,
        dropout,
        attn_dropout,
    ):
        super().__init__()
        self.n_out_steps = n_out_steps
        self.encoder_embedding = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = TransformerEncoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=int(d_model / n_heads),
            d_v=int(d_model / n_heads),
            d_ffn=d_ffn,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.transformer_decoder = TransformerDecoder(
            n_steps=n_steps,
            n_features=n_out_features,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=int(d_model / n_heads),
            d_v=int(d_model / n_heads),
            d_ffn=d_ffn,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.output = nn.Linear(d_model, n_out_features)

    def forward(self, X, forecasting_X):
        embedding = self.pos_encoding(self.encoder_embedding(X))
        encoding, _ = self.transformer_encoder(embedding)
        decoding = self.transformer_decoder(forecasting_X, encoding)
        output = self.output(decoding)
        return output[:, -self.n_out_steps :]


def train(model, train_dataloader, val_dataloader, test_dataloader):
    n_epochs = 100
    patience = 10
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    current_patience = patience
    best_loss = float("inf")
    for epoch in range(n_epochs):
        model.train()
        for idx, data in enumerate(train_dataloader):
            X, y = map(lambda x: x.to(args.device), data)
            optimizer.zero_grad()
            if "RNN" in model._get_name():
                predictions = model(X)
            else:
                predictions = model(X, y)
            loss = calc_mse(predictions, y[:, -n_forecasting_steps:])
            loss.backward()
            optimizer.step()

        model.eval()
        loss_collector = []
        with torch.no_grad():
            for idx, data in enumerate(val_dataloader):
                X, y = map(lambda x: x.to(args.device), data)
                if "RNN" in model._get_name():
                    predictions = model(X)
                else:
                    predictions = model(X, y)
                loss = calc_mse(predictions, y[:, -n_forecasting_steps:])
                loss_collector.append(loss.item())

        loss = np.asarray(loss_collector).mean()
        if best_loss > loss:
            current_patience = patience
            best_loss = loss
            best_model = model.state_dict()
        else:
            current_patience -= 1

        if current_patience == 0:
            break

    model.load_state_dict(best_model)
    model.eval()

    prediction_collector = []
    for idx, data in enumerate(test_dataloader):
        X, y = map(lambda x: x.to(args.device), data)
        if "RNN" in model._get_name():
            predictions = model(X)
        else:
            predictions = model(X, y)
        prediction_collector += predictions.cpu().tolist()

    prediction_collector = np.asarray(prediction_collector)
    return prediction_collector


def get_dataloaders(train_X, train_y, val_X, val_y, test_X, test_y, batch_size=128):
    train_set = LoadImputedData(train_X, train_y)
    val_set = LoadImputedData(val_X, val_y)
    test_set = LoadImputedData(test_X, test_y)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        help="device to run the model, e.g. cuda:0",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="the dataset name",
        required=True,
    )
    parser.add_argument(
        "--dataset_fold_path",
        type=str,
        help="the dataset fold path, where should include 3 H5 files train.h5, val.h5 and test.h5",
        required=True,
    )
    args = parser.parse_args()

    n_forecasting_steps = 5  # forecasting 5 steps ahead

    train_set_path = os.path.join(args.dataset_fold_path, "train.h5")
    val_set_path = os.path.join(args.dataset_fold_path, "val.h5")
    test_set_path = os.path.join(args.dataset_fold_path, "test.h5")
    with h5py.File(train_set_path, "r") as hf:
        pots_train_X = hf["X"][:]
        ori_train_X = hf["X_ori"][:]
    with h5py.File(val_set_path, "r") as hf:
        pots_val_X = hf["X"][:]
        ori_val_X = hf["X_ori"][:]
    with h5py.File(test_set_path, "r") as hf:
        pots_test_X = hf["X"][:]
        ori_test_X = hf["X_ori"][:]

    xgb_wo_metrics_collector = {"mae": [], "mse": [], "mre": []}
    xgb_metrics_collector = {"mae": [], "mse": [], "mre": []}
    rnn_metrics_collector = {"mae": [], "mse": [], "mre": []}
    transformer_metrics_collector = {"mae": [], "mse": [], "mre": []}

    imputed_data_path = os.path.join(
        args.dataset_fold_path,
        f"naive_imputation.h5",
    )
    imputed_data = load_dict_from_h5(imputed_data_path)
    for naive_method in ["mean", "median", "locf", "linear_interpolation"]:
        for n_round in range(5):
            train_X, val_X, test_X = (
                imputed_data["train"][naive_method],
                imputed_data["val"][naive_method],
                imputed_data["test"][naive_method],
            )

            train_y, val_y, test_y = (
                ori_train_X[:, :, -1:],
                ori_val_X[:, :, -1:],
                ori_test_X[:, :, -1:],
            )
            train_loader, val_loader, test_loader = get_dataloaders(
                train_X[:, :, :-1],
                train_y,
                val_X[:, :, :-1],
                val_y,
                test_X[:, :, :-1],
                test_y,
            )

            trans_train_X, trans_train_y = np.copy(train_X), np.copy(train_y)
            trans_val_X, trans_val_y = np.copy(val_X), np.copy(val_y)
            trans_test_X, trans_test_y = np.copy(test_X), np.copy(test_y)
            trans_val_X[:, -n_forecasting_steps:] = 0
            trans_val_y[:, :-n_forecasting_steps] = 0
            trans_test_X[:, -n_forecasting_steps:] = 0
            trans_test_y[:, :-n_forecasting_steps] = 0
            trans_train_loader, trans_val_loader, trans_test_loader = get_dataloaders(
                trans_train_X[:, :, :-1],
                trans_train_y,
                trans_val_X[:, :, :-1],
                trans_val_y,
                trans_test_X[:, :, :-1],
                trans_test_y,
            )

            set_random_seed(RANDOM_SEEDS[n_round])
            _, n_steps, n_features = train_X[:, :-n_forecasting_steps, :-1].shape
            n_in_flatten_features = n_steps * n_features
            n_out_flatten_features = np.product(
                train_y[:, -n_forecasting_steps:].shape[1:]
            )

            # XGBoost model with imputation
            xgb = XGBRegressor()
            xgb.fit(
                train_X[:, :-n_forecasting_steps, :-1].reshape(
                    -1, n_in_flatten_features
                ),
                train_y[:, -n_forecasting_steps:].reshape(-1, n_out_flatten_features),
                eval_set=[
                    (
                        val_X[:, :-n_forecasting_steps, :-1].reshape(
                            -1, n_in_flatten_features
                        ),
                        val_y[:, -n_forecasting_steps:].reshape(
                            -1, n_out_flatten_features
                        ),
                    )
                ],
                verbose=False,
            )
            predictions = xgb.predict(
                test_X[:, :-n_forecasting_steps, :-1].reshape(-1, n_in_flatten_features)
            )
            predictions = predictions.reshape(-1, n_forecasting_steps, 1)
            xgb_metrics_collector["mae"].append(
                calc_mae(predictions, test_y[:, -n_forecasting_steps:])
            )
            xgb_metrics_collector["mse"].append(
                calc_mse(predictions, test_y[:, -n_forecasting_steps:])
            )
            xgb_metrics_collector["mre"].append(
                calc_mre(predictions, test_y[:, -n_forecasting_steps:])
            )

            # RNN model
            simple_rnn_regressor = SimpleRNNForecaster(
                n_features=n_features,
                n_steps=n_steps,
                rnn_hidden_size=128,
                n_out_features=1,
                n_out_steps=n_forecasting_steps,
            )
            simple_rnn_regressor = simple_rnn_regressor.to(args.device)
            predictions = train(
                simple_rnn_regressor, train_loader, val_loader, test_loader
            )
            rnn_metrics_collector["mae"].append(
                calc_mae(predictions, test_y[:, -n_forecasting_steps:])
            )
            rnn_metrics_collector["mse"].append(
                calc_mse(predictions, test_y[:, -n_forecasting_steps:])
            )
            rnn_metrics_collector["mre"].append(
                calc_mre(predictions, test_y[:, -n_forecasting_steps:])
            )

            # Transformer model
            transformer_forecaster = TransformerForecaster(
                n_features=n_features,
                n_steps=n_steps + n_forecasting_steps,
                n_out_features=1,
                n_out_steps=n_forecasting_steps,
                n_layers=1,
                d_model=64,
                n_heads=2,
                d_ffn=128,
                dropout=0.1,
                attn_dropout=0,
            )
            transformer_forecaster = transformer_forecaster.to(args.device)
            predictions = train(
                transformer_forecaster,
                trans_train_loader,
                trans_val_loader,
                trans_test_loader,
            )
            transformer_metrics_collector["mae"].append(
                calc_mae(predictions, test_y[:, -n_forecasting_steps:])
            )
            transformer_metrics_collector["mse"].append(
                calc_mse(predictions, test_y[:, -n_forecasting_steps:])
            )
            transformer_metrics_collector["mre"].append(
                calc_mre(predictions, test_y[:, -n_forecasting_steps:])
            )

        logger.info(
            "\n"
            f"XGB (with {naive_method} imputation) regression "
            f"MAE: {np.mean(xgb_metrics_collector['mae']):.4f}±{np.std(xgb_metrics_collector['mae']):.4f}, "
            f"MSE: {np.mean(xgb_metrics_collector['mse']):.4f}±{np.std(xgb_metrics_collector['mse']):.4f}, "
            f"MRE: {np.mean(xgb_metrics_collector['mre']):.4f}±{np.std(xgb_metrics_collector['mre']):.4f}\n"
            f"RNN (with {naive_method} imputation) regression "
            f"MAE: {np.mean(rnn_metrics_collector['mae']):.4f}±{np.std(rnn_metrics_collector['mae']):.4f}, "
            f"MSE: {np.mean(rnn_metrics_collector['mse']):.4f}±{np.std(rnn_metrics_collector['mse']):.4f}, "
            f"MRE: {np.mean(rnn_metrics_collector['mre']):.4f}±{np.std(rnn_metrics_collector['mre']):.4f}\n"
            f"Transformer (with {naive_method} imputation) regression "
            f"MAE: {np.mean(transformer_metrics_collector['mae']):.4f}±{np.std(transformer_metrics_collector['mae']):.4f}, "
            f"MSE: {np.mean(transformer_metrics_collector['mse']):.4f}±{np.std(transformer_metrics_collector['mse']):.4f}, "
            f"MRE: {np.mean(transformer_metrics_collector['mre']):.4f}±{np.std(transformer_metrics_collector['mre']):.4f}\n"
        )
