"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import argparse
import os

import numpy as np
import torch
from pypots.data.saving import pickle_dump
from pypots.imputation import *
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed

from global_config import (
    TORCH_N_THREADS,
    RANDOM_SEEDS,
)
from hpo_results import HPO_RESULTS
from utils import get_datasets_path

SUPPORT_MODELS = {
    "Autoformer": Autoformer,
    "BRITS": BRITS,
    "Crossformer": Crossformer,
    "CSDI": CSDI,
    "DLinear": DLinear,
    "ETSformer": ETSformer,
    "FiLM": FiLM,
    "FreTS": FreTS,
    "GPVAE": GPVAE,
    "GRUD": GRUD,
    "Informer": Informer,
    "iTransformer": iTransformer,
    "Koopa": Koopa,
    "MICN": MICN,
    "MRNN": MRNN,
    "NonstationaryTransformer": NonstationaryTransformer,
    "PatchTST": PatchTST,
    "Pyraformer": Pyraformer,
    "SAITS": SAITS,
    "SCINet": SCINet,
    "StemGNN": StemGNN,
    "TimesNet": TimesNet,
    "Transformer": Transformer,
    "USGAN": USGAN,
}
SUPPORT_DATASETS = list(HPO_RESULTS.keys())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="the model name",
        required=True,
        choices=list(SUPPORT_MODELS.keys()),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="the dataset name",
        required=True,
        choices=SUPPORT_DATASETS,
    )
    parser.add_argument(
        "--dataset_fold_path",
        type=str,
        help="the dataset fold path, where should include 3 H5 files train.h5, val.h5 and test.h5",
        required=True,
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        help="the saving path of the model and logs",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run the model, e.g. cuda:0",
        required=True,
    )
    args = parser.parse_args()

    # set the number of threads for pytorch
    torch.set_num_threads(TORCH_N_THREADS)
    (
        train_set,
        val_set,
        test_X,
        test_X_ori,
        test_indicating_mask,
    ) = get_datasets_path(args.dataset_fold_path)

    mae_collector = []
    mse_collector = []
    mre_collector = []

    result_saving_path = os.path.join(args.saving_path, f"{args.model}_{args.dataset}")
    for n_round in range(1):
        set_random_seed(RANDOM_SEEDS[n_round])
        round_saving_path = os.path.join(result_saving_path, f"round_{n_round}")

        # get the hyperparameters and setup the model
        hyperparameters = HPO_RESULTS[args.dataset][args.model].copy()
        lr = hyperparameters.pop("lr")
        hyperparameters["device"] = args.device
        hyperparameters["saving_path"] = round_saving_path
        hyperparameters["model_saving_strategy"] = "best"
        if args.model == "USGAN":
            hyperparameters["G_optimizer"] = Adam(lr=lr)
            hyperparameters["D_optimizer"] = Adam(lr=lr)
        else:
            hyperparameters["optimizer"] = Adam(lr=lr)

        model = SUPPORT_MODELS[args.model](**hyperparameters)
        model.fit(train_set=train_set, val_set=val_set)

        test_set = {"X": test_X}
        if args.model == "CSDI":
            results = model.predict(test_set, n_sampling_times=10)
            imputation = results["imputation"].mean(axis=1)
        else:
            results = model.predict(test_set)
            imputation = results["imputation"]

        mae = calc_mae(imputation, test_X_ori, test_indicating_mask)
        mse = calc_mse(imputation, test_X_ori, test_indicating_mask)
        mre = calc_mre(imputation, test_X_ori, test_indicating_mask)
        mae_collector.append(mae)
        mse_collector.append(mse)
        mre_collector.append(mre)

        pickle_dump(
            imputation,
            os.path.join(round_saving_path, "imputation.pkl"),
        )
        logger.info(
            f"Round{n_round} - {args.model} on {args.dataset}: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}"
        )

    mean_mae, mean_mse, mean_mre = (
        np.mean(mae_collector),
        np.mean(mse_collector),
        np.mean(mre_collector),
    )
    std_mae, std_mse, std_mre = (
        np.std(mae_collector),
        np.std(mse_collector),
        np.std(mre_collector),
    )
    logger.info(
        f"Done! Final results:\n"
        f"Averaged {args.model} on {args.dataset}: "
        f"MAE={mean_mae:.4f} ± {std_mae}, "
        f"MSE={mean_mse:.4f} ± {std_mse}, "
        f"MRE={mean_mre:.4f} ± {std_mre}"
    )
