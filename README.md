## Description


This repository contains the code to reproduce the results of the paper entitled "Enriched Music Representations with Multiple Cross-modal Contrastive Learning", by Andres Ferraro, Xavier Favory, Konstantinos Drossos, Yuntae Kim, and Dmitry Bogdanov.

## Instructions to train the models

**Step 1**

Download the datasets from the [official website](https://arena.kakao.com/melon_dataset) and uncompress it.

**Step 2**

Generate the splits that will be used to train the models using the script: `scripts/create_dataset.py`. This scripts will first split the dataset, then create the CF embeddings with the songs in the training set and finally generate the hdf5 files with the train/val/test set.

**Step 3**

Pre-train w2v using the script: `train_genre_w2v.py`. Note that this step can be skipped since we already provide the `embedding_matrix_128.npy` file.

**Step 4**

In order to train the baseline models run the script `baseline_train.py` followed by the configuration file that indicates the encoder that has to be used:
 - To train the CF encoder, the following configiration file can be used: `python baseline_train.py configs/baseline_MF.json`
 - To train both CF and gnr encoders the following configuration file can be used: `python baseline_train.py configs/baseline_MF_gnr.json`

In order to train the models using contrastive learning run the script `single_train.py` and specify in the configuration file the encoder to user:
 - To train the CF encoder, the following configuration file can be used: `python single_train.py configs/contrastive_MF.json` 
 - To train the genre encoder, the following configuration file can be used: `python single_train.py configs/contrastive_gnr.json` 

Finally, to train the model that combines CF and genre data using contrastive learning run the script `train.py` scpecifing the configuration `configs/contrastive_MF_gnr.json`.

# Pre-trained models

The pre-trained models are provided in this repository under the folder `models`.

# Cite

Please citing the following publication: TODO
