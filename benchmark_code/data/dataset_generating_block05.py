"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from benchpots.datasets import (
    preprocess_beijing_air_quality,
    preprocess_italy_air_quality,
    preprocess_electricity_load_diagrams,
    preprocess_ett,
    preprocess_pems_traffic,
)
from pypots.utils.random import set_random_seed

from utils import organize_and_save

if __name__ == "__main__":
    set_random_seed(2024)
    pattern = "block"

    factor = 0.0055
    step = 24
    block_len = 6
    block_width = 6
    beijing_air_quality = preprocess_beijing_air_quality(
        rate=factor,
        n_steps=step,
        pattern=pattern,
        block_len=block_len,
        block_width=block_width,
    )
    organize_and_save(
        beijing_air_quality,
        f"generated_datasets/beijing_air_quality_rate{int(factor * 10):02d}_step{step}_{pattern}_blocklen{block_len}",
    )

    rate = 0.07
    step = 12
    block_len = 4
    block_width = 4
    italy_air_quality = preprocess_italy_air_quality(
        rate=rate,
        n_steps=step,
        pattern=pattern,
        block_len=block_len,
        block_width=block_width,
    )
    organize_and_save(
        italy_air_quality,
        f"generated_datasets/italy_air_quality_rate{int(rate * 10):02d}_step{step}_{pattern}_blocklen{block_len}",
    )

    rate = 0.002
    step = 96
    block_len = 8
    block_width = 8
    electricity_load_diagrams = preprocess_electricity_load_diagrams(
        rate=rate,
        n_steps=step,
        pattern=pattern,
        block_len=block_len,
        block_width=block_width,
    )
    organize_and_save(
        electricity_load_diagrams,
        f"generated_datasets/electricity_load_diagrams_rate{int(rate * 10):02d}_step{step}_{pattern}_blocklen{block_len}",
    )

    rate = 0.36
    step = 48
    block_len = 6
    block_width = 6
    ett = preprocess_ett(
        subset="ETTh1",
        rate=rate,
        n_steps=step,
        pattern=pattern,
        block_len=block_len,
        block_width=block_width,
    )
    organize_and_save(
        ett,
        f"generated_datasets/ett_rate{int(rate * 10):02d}_step{step}_{pattern}_blocklen{block_len}",
    )

    rate = 0.0008
    step = 24
    block_len = 6
    block_width = 6
    pems_traffic = preprocess_pems_traffic(
        file_path="/Users/wdu/Downloads/traffic.csv",
        rate=rate,
        n_steps=step,
        pattern=pattern,
        block_len=block_len,
        block_width=block_width,
    )
    organize_and_save(
        pems_traffic,
        f"generated_datasets/pems_traffic_rate{int(rate * 10):02d}_step{step}_{pattern}_blocklen{block_len}",
    )
