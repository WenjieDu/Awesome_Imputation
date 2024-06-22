# TSI-Bench 
The code scripts, configurations, and logs here are for TSI-Bench, 
the first comprehensive benchmark for time series imputation.

## ❖ Python Environment Creation
A proper Python environment is necessary to reproduce the results. 
Please ensure that all the below library requirements are satisfied.

```yaml
tsdb ==0.4
pygrinder ==0.6
benchpots ==0.1.1
pypots ==0.6
```

For Linux OS, it is able to create the environment with Conda by running `conda create -f conda_env.yml`.
For other OS, library version requirements can also be checked out in `conda_env.yml`.


## ❖ Datasets Generation
Please refer to [`data/README.md`](data/README.md).


## ❖ Results Reproduction
### Neural network training 
For example, to reproduce the results of SAITS on the dataset Pedestrian, please execute the following command.

```shell
nohup python train_model.py \
  --model SAITS \
  --dataset Pedestrian \
  --dataset_fold_path data/melbourne_pedestrian_rate01_step24_point \
  --saving_path results_point_rate01 \
  --device cuda:2 \
  > results_point_rate01/SAITS_pedestrian.log &
```

After the execution finished, please check out the logging information in the according `.log` file.

Additionally, as claimed in the paper, hyperparameters of all models get optimized by the tuning functionality in 
[PyPOTS](https://github.com/WenjieDu/PyPOTS). Hence, tuning configurations are available in the directory `PyPOTS_tuning_configs`.
If you'd like to explore this feature, please check out the details there.

### Naive methods
To obtain the results of the naive methods, check out the commands in the shell script `naive_imputation.sh`.


## ❖ Downstream Tasks
We're clean up the code and updating the scripts for the downstream tasks. Will release the code soon.