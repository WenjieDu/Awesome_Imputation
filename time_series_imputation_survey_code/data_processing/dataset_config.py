"""
Configs for dataset generation.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from pypots.utils.random import set_random_seed

# rate of artificial missing values added to the original data,
# for generating POTS dataset from complete time series or evaluating imputation models
ARTIFICIALLY_MISSING_RATE = 0.1

# random seed for numpy and pytorch
RANDOM_SEED = 2023
set_random_seed(RANDOM_SEED)
