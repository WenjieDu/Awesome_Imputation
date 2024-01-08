# Hyperparameter tuning configurations for PyPOTS and NNI

The hyperparameter tuning function in [PyPOTS](https://github.com/WenjieDu/PyPOTS) 
is implemented with [NNI](https://github.com/microsoft/nni).
The searching procedure is executed by NNI with unifed APIs provided in PyPOTS.

All the tuning configurations are stored in this directory. 
The NNI tuning configurations are in YAML files and the tuning spaces for model hyperparameters are defined in the JSON files.

For example, to start a NNI tuning experiment for SAITS on PhysioNet-2012 dataset, execute the following command: 

```shell
cd SAITS && nnictl create -c SAITS_searching_config.yml --port 8080
```

Note that 8080 is the default port, and a different port should be given for a new experiment if there're ones running.
