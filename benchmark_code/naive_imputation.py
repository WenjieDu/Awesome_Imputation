"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import argparse

import numpy as np
import pandas as pd
from pypots.data.saving import load_dict_from_h5, save_dict_into_h5
from pypots.imputation import LOCF, Mean, Median
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_fold_path",
        type=str,
        help="the dataset fold path, where should include 3 H5 files train.h5, val.h5 and test.h5",
        required=True,
    )

    args = parser.parse_args()
    naive_imputation = {}
    for subset in ["train", "val", "test"]:
        dataset = load_dict_from_h5(f"{args.dataset_fold_path}/{subset}.h5")
        mean_imputation = Mean().impute(dataset)
        median_imputation = Median().impute(dataset)
        locf_imputation = LOCF().impute(dataset)

        collector = []
        for i in dataset["X"]:
            filled_i = pd.DataFrame(i).interpolate(limit_direction="both")

            # fill still empty columns with mean
            # values in X are standardized, hence 0 below is actually the mean
            filled_i = filled_i.fillna(0)

            collector.append(filled_i.values)
        linear_interpolation = np.asarray(collector)

        if subset == "test":
            test_X, test_X_ori = dataset["X"], dataset["X_ori"]
            test_X_indicating_mask = np.isnan(test_X_ori) ^ np.isnan(test_X)
            mean_mae = calc_mae(
                mean_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            mean_mse = calc_mse(
                mean_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            mean_mre = calc_mre(
                mean_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            logger.info(
                f"mean imputation MAE: {mean_mae:.4f}, MSE: {mean_mse:.4f}, MRE: {mean_mre:.4f}"
            )
            median_mae = calc_mae(
                median_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            median_mse = calc_mse(
                median_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            median_mre = calc_mre(
                median_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            logger.info(
                f"median imputation MAE: {median_mae:.4f}, MSE: {median_mse:.4f}, MRE: {median_mre:.4f}"
            )
            locf_mae = calc_mae(
                locf_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            locf_mse = calc_mse(
                locf_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            locf_mre = calc_mre(
                locf_imputation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            logger.info(
                f"LOCF imputation MAE: {locf_mae:.4f}, MSE: {locf_mse:.4f}, MRE: {locf_mre:.4f}"
            )
            linear_interpolation_mae = calc_mae(
                linear_interpolation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            linear_interpolation_mse = calc_mse(
                linear_interpolation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            linear_interpolation_mre = calc_mre(
                linear_interpolation, np.nan_to_num(test_X_ori), test_X_indicating_mask
            )
            logger.info(
                f"linear interpolation imputation MAE: {linear_interpolation_mae:.4f}, "
                f"MSE: {linear_interpolation_mse:.4f}, MRE: {linear_interpolation_mre:.4f}"
            )

        naive_imputation[subset] = {
            "mean": mean_imputation,
            "median": median_imputation,
            "locf": locf_imputation,
            "linear_interpolation": linear_interpolation,
        }

    save_dict_into_h5(naive_imputation, args.dataset_fold_path, "naive_imputation.h5")
