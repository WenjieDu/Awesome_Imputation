"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from benchpots.preprocessing import (
    preprocess_physionet2012,
    preprocess_physionet2019,
    preprocess_beijing_air_quality,
    preprocess_italy_air_quality,
    preprocess_electricity_load_diagrams,
    preprocess_ett,
    preprocess_pems_traffic,
    preprocess_ucr_uea_datasets,
)
from pypots.data.saving import save_dict_into_h5
from pypots.utils.random import set_random_seed

# random seed for numpy and pytorch
RANDOM_SEED = 2024
set_random_seed(RANDOM_SEED)

# rate of artificial missing values added to the original data,
# for generating POTS dataset from complete time series or evaluating imputation models
rate = 0.1

# pattern of missing values
pattern = "point"

physionet_2012 = preprocess_physionet2012(
    rate=rate,
    pattern="point",
    subset="set-a",
    features=[
        "DiasABP",
        "HR",
        "Na",
        "Lactate",
        "NIDiasABP",
        "PaO2",
        "WBC",
        "pH",
        "Albumin",
        "ALT",
        "Glucose",
        "SaO2",
        "Temp",
        "AST",
        "Bilirubin",
        "HCO3",
        "BUN",
        "RespRate",
        "Mg",
        "HCT",
        "SysABP",
        "FiO2",
        "K",
        "GCS",
        "Cholesterol",
        "NISysABP",
        "TroponinT",
        "MAP",
        "TroponinI",
        "PaCO2",
        "Platelets",
        "Urine",
        "NIMAP",
        "Creatinine",
        "ALP",
    ],
)
physionet_2012_train = {
    "X": physionet_2012["train_X"],
}
physionet_2012_val = {
    "X": physionet_2012["val_X"],
    "X_ori": physionet_2012["val_X_ori"],
}
physionet_2012_test = {
    "X": physionet_2012["test_X"],
    "X_ori": physionet_2012["test_X_ori"],
}
save_dict_into_h5(
    physionet_2012_train,
    "physionet_2012_rate01_point",
    "train.h5",
)
save_dict_into_h5(
    physionet_2012_val,
    "physionet_2012_rate01_point",
    "val.h5",
)
save_dict_into_h5(
    physionet_2012_test,
    "physionet_2012_rate01_point",
    "test.h5",
)
print("\n\n\n")

physionet_2019 = preprocess_physionet2019(rate=rate, pattern="point", subset="all")
physionet_2019_train = {
    "X": physionet_2019["train_X"],
}
physionet_2019_val = {
    "X": physionet_2019["val_X"],
    "X_ori": physionet_2019["val_X_ori"],
}
physionet_2019_test = {
    "X": physionet_2019["test_X"],
    "X_ori": physionet_2019["test_X_ori"],
}
save_dict_into_h5(
    physionet_2019_train,
    "physionet_2019_rate01_point",
    "train.h5",
)
save_dict_into_h5(
    physionet_2019_val,
    "physionet_2019_rate01_point",
    "val.h5",
)
save_dict_into_h5(
    physionet_2019_test,
    "physionet_2019_rate01_point",
    "test.h5",
)
print("\n\n\n")

step = 24
beijing_air_quality = preprocess_beijing_air_quality(
    rate=rate, n_steps=step, pattern=pattern
)
beijing_air_quality_train = {
    "X": beijing_air_quality["train_X"],
}
beijing_air_quality_val = {
    "X": beijing_air_quality["val_X"],
    "X_ori": beijing_air_quality["val_X_ori"],
}
beijing_air_quality_test = {
    "X": beijing_air_quality["test_X"],
    "X_ori": beijing_air_quality["test_X_ori"],
}
save_dict_into_h5(
    beijing_air_quality_train,
    f"beijing_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "train.h5",
)
save_dict_into_h5(
    beijing_air_quality_val,
    f"beijing_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "val.h5",
)
save_dict_into_h5(
    beijing_air_quality_test,
    f"beijing_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "test.h5",
)
print("\n\n\n")

step = 12
italy_air_quality = preprocess_italy_air_quality(
    rate=rate, n_steps=step, pattern=pattern
)
italy_air_quality_train = {
    "X": italy_air_quality["train_X"],
}
italy_air_quality_val = {
    "X": italy_air_quality["val_X"],
    "X_ori": italy_air_quality["val_X_ori"],
}
italy_air_quality_test = {
    "X": italy_air_quality["test_X"],
    "X_ori": italy_air_quality["test_X_ori"],
}
save_dict_into_h5(
    italy_air_quality_train,
    f"italy_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "train.h5",
)
save_dict_into_h5(
    italy_air_quality_val,
    f"italy_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "val.h5",
)
save_dict_into_h5(
    italy_air_quality_test,
    f"italy_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "test.h5",
)
print("\n\n\n")

step = 96
electricity_load_diagrams = preprocess_electricity_load_diagrams(
    rate=rate,
    n_steps=step,
    pattern=pattern,
)
electricity_load_diagrams_train = {
    "X": electricity_load_diagrams["train_X"],
}
electricity_load_diagrams_val = {
    "X": electricity_load_diagrams["val_X"],
    "X_ori": electricity_load_diagrams["val_X_ori"],
}
electricity_load_diagrams_test = {
    "X": electricity_load_diagrams["test_X"],
    "X_ori": electricity_load_diagrams["test_X_ori"],
}
save_dict_into_h5(
    electricity_load_diagrams_train,
    f"electricity_load_diagrams_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "train.h5",
)
save_dict_into_h5(
    electricity_load_diagrams_val,
    f"electricity_load_diagrams_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "val.h5",
)
save_dict_into_h5(
    electricity_load_diagrams_test,
    f"electricity_load_diagrams_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "test.h5",
)
print("\n\n\n")

step = 48
ett = preprocess_ett(
    set_name="ETTh1",
    rate=rate,
    n_steps=step,
    pattern=pattern,
)
ett_train = {
    "X": ett["train_X"],
}
ett_val = {
    "X": ett["val_X"],
    "X_ori": ett["val_X_ori"],
}
ett_test = {
    "X": ett["test_X"],
    "X_ori": ett["test_X_ori"],
}
save_dict_into_h5(
    ett_train,
    f"ett_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "train.h5",
)
save_dict_into_h5(
    ett_val,
    f"ett_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "val.h5",
)
save_dict_into_h5(
    ett_test,
    f"ett_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "test.h5",
)
print("\n\n\n")

step = 24
pems_traffic = preprocess_pems_traffic(
    file_path="traffic.csv",
    rate=rate,
    n_steps=step,
    pattern=pattern,
)
pems_traffic_train = {
    "X": pems_traffic["train_X"],
}
pems_traffic_val = {
    "X": pems_traffic["val_X"],
    "X_ori": pems_traffic["val_X_ori"],
}
pems_traffic_test = {
    "X": pems_traffic["test_X"],
    "X_ori": pems_traffic["test_X_ori"],
}
save_dict_into_h5(
    pems_traffic_train,
    f"pems_traffic_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "train.h5",
)
save_dict_into_h5(
    pems_traffic_val,
    f"pems_traffic_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "val.h5",
)
save_dict_into_h5(
    pems_traffic_test,
    f"pems_traffic_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "test.h5",
)
print("\n\n\n")

step = 24
melbourne_pedestrian = preprocess_ucr_uea_datasets(
    "ucr_uea_MelbournePedestrian",
    rate=rate,
)
melbourne_pedestrian_train = {
    "X": melbourne_pedestrian["train_X"],
}
melbourne_pedestrian_val = {
    "X": melbourne_pedestrian["val_X"],
    "X_ori": melbourne_pedestrian["val_X_ori"],
}
melbourne_pedestrian_test = {
    "X": melbourne_pedestrian["test_X"],
    "X_ori": melbourne_pedestrian["test_X_ori"],
}
save_dict_into_h5(
    melbourne_pedestrian_train,
    f"melbourne_pedestrian_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "train.h5",
)
save_dict_into_h5(
    melbourne_pedestrian_val,
    f"melbourne_pedestrian_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "val.h5",
)
save_dict_into_h5(
    melbourne_pedestrian_test,
    f"melbourne_pedestrian_rate{int(rate * 10):02d}_step{step}_{pattern}",
    "test.h5",
)
