# Data generation

Run the below commands to generate datasets for experiments.
Note that, for PeMS traffic dataset, you have to put the `traffic.csv` file under the current directory.
You can download it from https://pems.dot.ca.gov. For other dataset, they are integrated into `TSDB` and can be directly used. 

```shell
python dataset_generating_point01.py
python dataset_generating_point05.py
python dataset_generating_point09.py
python dataset_generating_subseq05.py
python dataset_generating_block05.py
```