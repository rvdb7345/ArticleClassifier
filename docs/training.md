Training
========

This ReadMe explains how to train and evaluate a GNN on the data using the command line.

### Before training a model

* make sure the environment is built from the requirements.txt file and activated.
* build the datasets as described in the data_preparation folder

## Training a model

Training a new model is very straightforward. After setting the preferred settings as per the 'settings' page, you can
run:

```commandline
python main.py --run_model 'canary' 'GAT'
```

## Running a trained model

In case you want to rerun a trained model, you can add the id of the model:

```commandline
python main.py --run_model 'litcovid' 'GAT' --model_id model_id
```
