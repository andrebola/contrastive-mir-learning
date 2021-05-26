## Description


This repository contains the code to reproduce the results of the paper entitled "Enriched Music Representations with Multiple Cross-modal Contrastive Learning", by Andres Ferraro, Xavier Favory, Konstantinos Drossos, Yuntae Kim, and Dmitry Bogdanov.

# Demo

The following [demo](http://fonil.mtg.upf.edu/) shows how the embeddings can be used for similarity. For each playlist we take 200 random tracks and connect only the ones that are more similar to the tracks in the playlist. Note that the demo only uses 10% of the tracks in the dataset, therefore, some playlists don't have tracks.


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

Please cite the following publication: 

> A. Ferraro, X. Favory, K. Drossos, Y. Kim and D. Bogdanov, "Enriched Music Representations With Multiple Cross-Modal Contrastive Learning," in IEEE Signal Processing Letters, vol. 28, pp. 733-737, 2021, doi: 10.1109/LSP.2021.3071082.

```
@article {ferraro2021spl,
    author = "Ferraro, Andres and Favory, Xavier and Drossos, Konstantinos and Kim, Yuntae and Bogdanov, Dmitry",
    title = "Enriched Music Representations with Multiple Cross-modal Contrastive Learning",
    journal={IEEE Signal Processing Letters}, 
    volume={28},
    number={},
    pages={733-737},
    doi={10.1109/LSP.2021.3071082}},
    year = "2021"
}
```
