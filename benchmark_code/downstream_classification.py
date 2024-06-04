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
import torch.nn.functional as F
from pypots.data.saving import pickle_load
from pypots.nn.modules.transformer import TransformerEncoder, PositionalEncoding
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_binary_classification_metrics
from pypots.utils.random import set_random_seed
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier

from global_config import RANDOM_SEEDS


class LoadImputedDataAndLabel(Dataset):
    def __init__(self, imputed_data, labels):
        self.imputed_data = imputed_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.imputed_data[idx]).to(torch.float32),
            torch.tensor(self.labels[idx]).to(torch.long),
        )


class SimpleRNNClassification(torch.nn.Module):
    def __init__(self, n_features, rnn_hidden_size, n_classes):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            n_features,
            hidden_size=rnn_hidden_size,
            batch_first=True,
        )
        self.fcn = torch.nn.Linear(rnn_hidden_size, n_classes)

    def forward(self, data):
        hidden_states, _ = self.rnn(data)
        logits = self.fcn(hidden_states[:, -1, :])
        prediction_probabilities = torch.sigmoid(logits)
        return prediction_probabilities


class TransformerClassification(torch.nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        d_model,
        n_heads,
        d_ffn,
        dropout,
        attn_dropout,
        n_classes,
    ):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
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
        self.output = nn.Linear(d_model * n_steps, n_classes)

    def forward(self, data):
        bz = data.shape[0]
        embedding = self.pos_encoding(self.embedding(data))
        encoding, _ = self.transformer_encoder(embedding)
        encoding = encoding.reshape(bz, -1)
        logits = self.output(encoding)
        prediction_probabilities = torch.sigmoid(logits)
        return prediction_probabilities


def train(model, train_dataloader, val_dataloader):
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
            probabilities = model(X)
            loss = F.cross_entropy(probabilities, y.reshape(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        loss_collector = []
        with torch.no_grad():
            for idx, data in enumerate(val_dataloader):
                X, y = map(lambda x: x.to(args.device), data)
                probabilities = model(X)
                loss = F.cross_entropy(probabilities, y.reshape(-1))
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

    probability_collector = []
    for idx, data in enumerate(test_loader):
        X, y = map(lambda x: x.to(args.device), data)
        probabilities = model.forward(X)
        probability_collector += probabilities.cpu().tolist()

    probability_collector = np.asarray(probability_collector)
    return probability_collector


def get_dataloaders(train_X, train_y, val_X, val_y, test_X, test_y, batch_size=128):
    train_set = LoadImputedDataAndLabel(train_X, train_y)
    val_set = LoadImputedDataAndLabel(val_X, val_y)
    test_set = LoadImputedDataAndLabel(test_X, test_y)
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
        "--model",
        type=str,
        help="the model name",
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
    parser.add_argument(
        "--n_classes",
        type=int,
        help="the number of classes",
        required=True,
    )
    parser.add_argument(
        "--model_result_parent_fold",
        type=str,
        help="the parent fold of the model results, where should include the folds of 5 rounds",
        required=True,
    )
    args = parser.parse_args()

    train_set_path = os.path.join(args.dataset_fold_path, "train.h5")
    val_set_path = os.path.join(args.dataset_fold_path, "val.h5")
    test_set_path = os.path.join(args.dataset_fold_path, "test.h5")
    with h5py.File(train_set_path, "r") as hf:
        pots_train_X = hf["X"][:]
        train_y = hf["y"][:]
    with h5py.File(val_set_path, "r") as hf:
        pots_val_X = hf["X"][:]
        val_y = hf["y"][:]
    with h5py.File(test_set_path, "r") as hf:
        pots_test_X = hf["X"][:]
        test_y = hf["y"][:]

    if args.dataset == "Pedestrian":
        # Pedestrian dataset has 10 classes with label from 1 to 10, we need to convert it to 0 to 9
        train_y, val_y, test_y = train_y - 1, val_y - 1, test_y - 1

    train_X_collector = []
    val_X_collector = []
    test_X_collector = []

    for n_round in range(5):
        imputed_data_path = os.path.join(
            args.model_result_parent_fold,
            f"round_{n_round}/imputation.pkl",
        )
        imputed_data = pickle_load(imputed_data_path)
        _train_X, _val_X, _test_X = (
            imputed_data["train_set_imputation"],
            imputed_data["val_set_imputation"],
            imputed_data["test_set_imputation"],
        )
        train_X_collector.append(_train_X)
        val_X_collector.append(_val_X)
        test_X_collector.append(_test_X)
    train_X, val_X, test_X = (
        np.mean(np.stack(train_X_collector), axis=0),
        np.mean(np.stack(val_X_collector), axis=0),
        np.mean(np.stack(test_X_collector), axis=0),
    )

    xgb_wo_pr_auc_collector = []
    xgb_wo_roc_auc_collector = []
    xgb_pr_auc_collector = []
    xgb_roc_auc_collector = []
    rnn_pr_auc_collector = []
    rnn_roc_auc_collector = []
    transformer_pr_auc_collector = []
    transformer_roc_auc_collector = []

    train_loader, val_loader, test_loader = get_dataloaders(
        train_X, train_y, val_X, val_y, test_X, test_y
    )
    for n_round in range(5):
        set_random_seed(RANDOM_SEEDS[n_round])
        # XGBoost model without imputation
        n_flatten_features = np.product(train_X.shape[1:])
        xgb = XGBClassifier()
        xgb.fit(
            pots_train_X.reshape(-1, n_flatten_features),
            train_y,
            eval_set=[(pots_val_X.reshape(-1, n_flatten_features), val_y)],
            verbose=False,
        )
        proba_predictions = xgb.predict_proba(
            pots_test_X.reshape(-1, n_flatten_features)
        )
        if args.n_classes == 2:
            classification_metrics = calc_binary_classification_metrics(
                proba_predictions, test_y
            )
            pr_auc, roc_auc = (
                classification_metrics["pr_auc"],
                classification_metrics["roc_auc"],
            )
        else:
            pr_auc, roc_auc = None, None
        xgb_wo_pr_auc_collector.append(pr_auc)
        xgb_wo_roc_auc_collector.append(roc_auc)

        # XGBoost model without imputation
        xgb = XGBClassifier()
        xgb.fit(
            train_X.reshape(-1, n_flatten_features),
            train_y,
            eval_set=[(val_X.reshape(-1, n_flatten_features), val_y)],
            verbose=False,
        )
        proba_predictions = xgb.predict_proba(test_X.reshape(-1, n_flatten_features))
        if args.n_classes == 2:
            classification_metrics = calc_binary_classification_metrics(
                proba_predictions, test_y
            )
            pr_auc, roc_auc = (
                classification_metrics["pr_auc"],
                classification_metrics["roc_auc"],
            )
        else:
            pr_auc, roc_auc = None, None
        xgb_pr_auc_collector.append(pr_auc)
        xgb_roc_auc_collector.append(roc_auc)

        # RNN model
        simple_rnn_classifier = SimpleRNNClassification(
            n_features=train_X.shape[-1],
            rnn_hidden_size=128,
            n_classes=args.n_classes,
        )
        simple_rnn_classifier = simple_rnn_classifier.to(args.device)
        proba_predictions = train(simple_rnn_classifier, train_loader, val_loader)
        if args.n_classes == 2:
            classification_metrics = calc_binary_classification_metrics(
                proba_predictions, test_y
            )
            pr_auc, roc_auc = (
                classification_metrics["pr_auc"],
                classification_metrics["roc_auc"],
            )
        else:
            pr_auc, roc_auc = None, None
        rnn_pr_auc_collector.append(pr_auc)
        rnn_roc_auc_collector.append(roc_auc)

        # Transformer model
        transformer_classifier = TransformerClassification(
            n_steps=train_X.shape[1],
            n_features=train_X.shape[2],
            n_layers=1,
            d_model=64,
            n_heads=2,
            d_ffn=128,
            dropout=0.1,
            attn_dropout=0,
            n_classes=args.n_classes,
        )
        transformer_classifier = transformer_classifier.to(args.device)
        proba_predictions = train(transformer_classifier, train_loader, val_loader)
        if args.n_classes == 2:
            classification_metrics = calc_binary_classification_metrics(
                proba_predictions, test_y
            )
            pr_auc, roc_auc = (
                classification_metrics["pr_auc"],
                classification_metrics["roc_auc"],
            )
        else:
            pr_auc, roc_auc = None, None
        transformer_pr_auc_collector.append(pr_auc)
        transformer_roc_auc_collector.append(roc_auc)

    logger.info(
        "\n"
        f"XGB without imputation PR_AUC: {np.mean(xgb_wo_pr_auc_collector):.4f}±{np.std(xgb_wo_pr_auc_collector):.4f}, "
        f"ROC_AUC: {np.mean(xgb_wo_roc_auc_collector):.4f}±{np.std(xgb_wo_roc_auc_collector):.4f}\n"
        f"XGB with {args.model} imputation PR_AUC: {np.mean(xgb_pr_auc_collector):.4f}±{np.std(xgb_pr_auc_collector):.4f}, "
        f"ROC_AUC: {np.mean(xgb_roc_auc_collector):.4f}±{np.std(xgb_roc_auc_collector):.4f}\n"
        f"RNN with {args.model} imputation PR_AUC: {np.mean(rnn_pr_auc_collector):.4f}±{np.std(rnn_pr_auc_collector):.4f}, "
        f"ROC_AUC: {np.mean(rnn_roc_auc_collector):.4f}±{np.std(rnn_roc_auc_collector):.4f}\n"
        f"Transformer with {args.model} imputation PR_AUC: {np.mean(transformer_pr_auc_collector):.4f}±{np.std(transformer_pr_auc_collector):.4f}, "
        f"ROC_AUC: {np.mean(transformer_roc_auc_collector):.4f}±{np.std(transformer_roc_auc_collector):.4f}\n"
    )
