Preparing data
==============

This ReadMe explains how to prepare your data sources from the raw CARs to finetuned embedding text and graph networks. 

Processing CAR data
^^^^^^^^^^^^^^^^^^^
From the CAR data we create a single CSV file to work with, extracting the data that we want.



Creating graph networks
^^^^^^^^^^^^^^^^^^^^^^^
To train GNNs, we need graph networks. We build them using the create_author_networks.py and create_keyword_networks.py files.

* make sure the environment is built from the requirements.txt file.
* build the datasets as described in the data_preparation folder


Training a model
^^^^^^^^^^^^^^^^

