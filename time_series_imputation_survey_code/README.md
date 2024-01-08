# Code for the Time Series Imputation Survey 
The scripts and configurations used in the work are all put here.


## ❖ Python environment creation
A proper Python environment is necessary to reproduce the results. 
Please ensure that all the below library requirements are satisfied.

```yaml
pypots >=0.3
tsdb >=0.2
pygrinder >=0.4
```

For Linux OS, it is able to create the environment with Conda by running `conda create -f conda_env.yml`.
For other OS, library version requirements can also be checked out in `conda_env.yml`.


## ❖ Data generation
The scripts for generating three datasets used in this work are in the directory `data_processing`. 
To generate the preprocessed datasets, please run the shell script `generate_datasets.sh` or 
execute the below commands:

```shell
# generate PhysioNet2012 dataset
python data_processing/gene_physionet_2012.py

# generate Air dataset
python data_processing/gene_air_quality.py

# generate ETTm1 dataset
python data_processing/gene_ettm1.py
```


## ❖ Model training and results reproduction
```shell
# reproduce the results on the dataset PhysioNet2012
nohup python train_models_for_physionet2012.py > physionet2012.log&

# reproduce the results on the dataset Air
nohup python train_models_for_air.py > air.log&

# reproduce the results on the dataset ETTm1
nohup python train_models_for_ettm1.py > ettm1.log&
```

After all execution finished, please check out all logging information in the according `.log` files.

Additionally, as claimed in the paper, hyperparameters of all models get optimized by the tuning functionality in 
[PyPOTS](https://github.com/WenjieDu/PyPOTS). Hence, tuning configurations are available in the directory `PyPOTS_tuning_configs`.
If you'd like to explore this feature, please check out the details there.


## ❖ Downstream classification
After running `train_models_for_physionet2012.py`, all models' imputation results are persisted under according folders.
To obtain the simple RNN's classification results on PhysioNet2012, please execute the script `downstream_classification.py`.