"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from benchpots.datasets import (
    preprocess_physionet2012,
    preprocess_physionet2019,
    preprocess_beijing_air_quality,
    preprocess_italy_air_quality,
    preprocess_electricity_load_diagrams,
    preprocess_ett,
    preprocess_pems_traffic,
    preprocess_ucr_uea_datasets,
)
from pypots.utils.random import set_random_seed

from utils import organize_and_save

if __name__ == "__main__":
    set_random_seed(2024)
    rate = 0.1
    pattern = "point"

    physionet_2012 = preprocess_physionet2012(
        subset="set-a",
        rate=rate,
        pattern="point",
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
    organize_and_save(physionet_2012, "generated_datasets/physionet_2012_rate01_point")

    step = 24
    beijing_air_quality = preprocess_beijing_air_quality(
        rate=rate,
        n_steps=step,
        pattern=pattern,
    )
    organize_and_save(
        beijing_air_quality,
        f"generated_datasets/beijing_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}",
    )

    step = 12
    italy_air_quality = preprocess_italy_air_quality(
        rate=rate,
        n_steps=step,
        pattern=pattern,
    )
    organize_and_save(
        italy_air_quality,
        f"generated_datasets/italy_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}",
    )

    step = 96
    electricity_load_diagrams = preprocess_electricity_load_diagrams(
        rate=rate,
        n_steps=step,
        pattern=pattern,
    )
    organize_and_save(
        electricity_load_diagrams,
        f"generated_datasets/electricity_load_diagrams_rate{int(rate * 10):02d}_step{step}_{pattern}",
    )

    step = 48
    ett = preprocess_ett(
        subset="ETTh1",
        rate=rate,
        n_steps=step,
        pattern=pattern,
    )
    organize_and_save(
        ett, f"generated_datasets/ett_rate{int(rate * 10):02d}_step{step}_{pattern}"
    )

    step = 24
    pems_traffic = preprocess_pems_traffic(
        rate=rate,
        n_steps=step,
        pattern=pattern,
    )
    organize_and_save(
        pems_traffic,
        f"generated_datasets/pems_traffic_rate{int(rate * 10):02d}_step{step}_{pattern}",
    )

    step = 24
    melbourne_pedestrian = preprocess_ucr_uea_datasets(
        dataset_name="ucr_uea_MelbournePedestrian",
        rate=rate,
    )
    organize_and_save(
        melbourne_pedestrian,
        f"generated_datasets/melbourne_pedestrian_rate{int(rate * 10):02d}_step{step}_{pattern}",
    )

    physionet_2019 = preprocess_physionet2019(
        subset="training_setA",
        rate=rate,
        pattern="point",
    )
    organize_and_save(
        physionet_2019, "generated_datasets/physionet_2019_rate01_step48_point"
    )
