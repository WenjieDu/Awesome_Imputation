"""
A simple RNN classifier for downstream classification task on imputed Physionet 2012 dataset.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from pypots.data.saving import pickle_load
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_binary_classification_metrics
from pypots.utils.random import set_random_seed
from torch.utils.data import Dataset, DataLoader

from global_config import RESULTS_SAVING_PATH, MAX_N_EPOCHS, DEVICE, RANDOM_SEEDS


class LoadImputedDataAndLabel(Dataset):
    def __init__(self, imputed_data, labels):
        self.imputed_data = imputed_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.imputed_data[idx]).to(torch.float32),
            torch.tensor(self.labels[idx]).to(torch.float32),
        )


class SimpleRNNClassification(torch.nn.Module):
    def __init__(self, feature_num, rnn_hidden_size, class_num):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            feature_num,
            hidden_size=rnn_hidden_size,
            batch_first=True,
        )
        self.fcn = torch.nn.Linear(rnn_hidden_size, class_num)

    def forward(self, data):
        hidden_states, _ = self.rnn(data)
        logits = self.fcn(hidden_states[:, -1, :])
        prediction_probabilities = torch.sigmoid(logits)
        return prediction_probabilities


def train(model, train_dataloader, val_dataloader, optimizer):
    patience = 20
    current_patience = patience
    best_loss = 1000
    for epoch in range(MAX_N_EPOCHS):
        model.train()
        for idx, data in enumerate(train_dataloader):
            X, y = map(lambda x: x.to(DEVICE), data)
            optimizer.zero_grad()
            probabilities = model(X)
            loss = F.binary_cross_entropy(probabilities, y.reshape(-1, 1))
            loss.backward()
            optimizer.step()

        model.eval()
        loss_collector = []
        with torch.no_grad():
            for idx, data in enumerate(val_dataloader):
                X, y = map(lambda x: x.to(DEVICE), data)
                probabilities = model(X)
                loss = F.binary_cross_entropy(probabilities, y.reshape(-1, 1))
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

    return best_model


def get_dataloaders(train_X, train_y, val_X, val_y, test_X, test_y, batch_size=128):
    train_set = LoadImputedDataAndLabel(train_X, train_y)
    val_set = LoadImputedDataAndLabel(val_X, val_y)
    test_set = LoadImputedDataAndLabel(test_X, test_y)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    saving_dir = "data/physionet_2012"
    train_set_path = os.path.join(saving_dir, "train.h5")
    val_set_path = os.path.join(saving_dir, "val.h5")
    test_set_path = os.path.join(saving_dir, "test.h5")
    with h5py.File(train_set_path, "r") as hf:
        train_y = hf["y"][:]
    with h5py.File(val_set_path, "r") as hf:
        val_y = hf["y"][:]
    with h5py.File(test_set_path, "r") as hf:
        test_y = hf["y"][:]

    for method_name in [
        "Mean",
        "Median",
        "LOCF",
        "MRNN",
        "GPVAE",
        "BRITS",
        "USGAN",
        "CSDI",
        "TimesNet",
        "Transformer",
        "SAITS",
    ]:
        pr_auc_collector = []
        roc_auc_collector = []

        train_X_collector = []
        val_X_collector = []
        test_X_collector = []

        for n_round in range(5):
            imputed_data_path = os.path.join(
                RESULTS_SAVING_PATH,
                f"round_{n_round}/{method_name}_physionet2012/imputation.pkl",
            )
            [_train_X, _val_X, _test_X] = pickle_load(imputed_data_path)
            train_X_collector.append(_train_X)
            val_X_collector.append(_val_X)
            test_X_collector.append(_test_X)
        train_X, val_X, test_X = (
            np.mean(np.stack(train_X_collector), axis=0),
            np.mean(np.stack(val_X_collector), axis=0),
            np.mean(np.stack(test_X_collector), axis=0),
        )

        for n_round in range(5):
            set_random_seed(RANDOM_SEEDS[n_round])
            train_loader, val_loader, test_loader = get_dataloaders(
                train_X, train_y, val_X, val_y, test_X, test_y
            )
            simple_rnn_classifier = SimpleRNNClassification(
                feature_num=37, rnn_hidden_size=128, class_num=1
            )
            optimizer = torch.optim.Adam(simple_rnn_classifier.parameters(), 1e-3)
            if "cuda" in DEVICE:
                simple_rnn_classifier = simple_rnn_classifier.cuda()
            # training and validating
            best_model = train(
                simple_rnn_classifier, train_loader, val_loader, optimizer
            )
            simple_rnn_classifier.load_state_dict(best_model)

            # testing stage
            probability_collector, label_collector = [], []
            for idx, data in enumerate(test_loader):
                X, y = map(lambda x: x.to(DEVICE), data)
                probabilities = simple_rnn_classifier.forward(X)
                probability_collector += probabilities.cpu().tolist()
                label_collector += y.cpu().tolist()
            probability_collector = np.asarray(probability_collector)
            label_collector = np.asarray(label_collector)
            classification_metrics = calc_binary_classification_metrics(
                probability_collector, label_collector
            )
            pr_auc_collector.append(classification_metrics["pr_auc"])
            roc_auc_collector.append(classification_metrics["roc_auc"])

        logger.info(
            f"RNN on {method_name} imputation PR_AUC: {np.mean(pr_auc_collector):.4f}±{np.std(pr_auc_collector):.4f}, "
            f"ROC_AUC: {np.mean(roc_auc_collector):.4f}±{np.std(roc_auc_collector):.4f}"
        )

# 2024-01-05 14:51:35 [INFO]: RNN on Mean imputation PR_AUC: 0.4337±0.0158, ROC_AUC: 0.8134±0.0087
# 2024-01-05 14:52:17 [INFO]: RNN on Median imputation PR_AUC: 0.4338±0.0176, ROC_AUC: 0.8081±0.0142
# 2024-01-05 14:52:44 [INFO]: RNN on LOCF imputation PR_AUC: 0.4250±0.0151, ROC_AUC: 0.8041±0.0068
# 2024-01-05 14:53:17 [INFO]: RNN on MRNN imputation PR_AUC: 0.4242±0.0219, ROC_AUC: 0.8069±0.0147
# 2024-01-05 14:53:46 [INFO]: RNN on GPVAE imputation PR_AUC: 0.3842±0.0180, ROC_AUC: 0.7879±0.0082
# 2024-01-05 14:54:17 [INFO]: RNN on BRITS imputation PR_AUC: 0.4277±0.0166, ROC_AUC: 0.8211±0.0079
# 2024-01-05 14:54:47 [INFO]: RNN on USGAN imputation PR_AUC: 0.4313±0.0171, ROC_AUC: 0.8144±0.0099
# 2024-01-05 14:55:11 [INFO]: RNN on CSDI imputation PR_AUC: 0.4327±0.0084, ROC_AUC: 0.8108±0.0048
# 2024-01-05 14:55:35 [INFO]: RNN on TimesNet imputation PR_AUC: 0.4058±0.0121, ROC_AUC: 0.7872±0.0128
# 2024-01-05 14:56:05 [INFO]: RNN on Transformer imputation PR_AUC: 0.4455±0.0314, ROC_AUC: 0.8072±0.0182
# 2024-01-05 14:56:34 [INFO]: RNN on SAITS imputation PR_AUC: 0.4549±0.0158, ROC_AUC: 0.8222±0.0019
