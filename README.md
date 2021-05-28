# Candidate-Free Taxonomy Enrichment
Candidate-free taxonomy enrichment NLP/DL project

## Team Members
* Timotei Ardelean
* Andreea Dogaru
* Gabriel Rozzonelli
* Alsu Vakhitova

## Repository Overview
configs/ - configuration files to be used with train.py to train different combinations of models / datasets / optimizers / schedulers and their parameters
* [`configs/`](configs/) - configuration files to be used with `train.py` to train different models and set hyperparameters
* [`data/`](data/) - contains logic for downloading and processing the dataset
* [`metrics/`](metrics/) - contains implementation of PrecisionK@k, MRR and MAP
* [`models/`](models/) - each of the proposed models is implemented in a separate file
* [`notebooks/`](notebooks/) - the main notebook clones the repository and has the template the code for training and testing based on a config file
* [`notebooks/`](systems/) - accommodates high order modules which contain a Model and implement inference, optimization, and logging
* 👉[`train.py`](train.py) - entry point for training a system based on a configuration file
* 👉[`evaluate.py`](evaluate.py) - entry point for evaluating a system, can be used to obtain the metrics for any split of the dataset


### Training setup 
The training process can be started with the provided script:
```shell
python train.py experiment=<config_name>
```
Where config_name can be one of the config files provided (`fixed`|`bert`|`bert_gat`) or a custom .yaml file following the same template. Any part of the configuration can also be overwritten from CLI. For example, to train for a different number of epochs one can write:
```shell
python train.py experiment=<config_name> experiment.trainer_args.max_epochs=50
```
