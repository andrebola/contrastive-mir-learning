"""
This script is used to create the HDF5 training and validation dataset files.
"""
import sys
sys.path.append('..')
import os
import json
import pickle
import h5py
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="1" #"0,1,2,3"

from scipy import sparse
from lightfm import LightFM

from utils import ProgressBar
NUM_BANDS = 48
NUM_FRAMES = 1300
NUM_TAGS = 100
#NUM_TAGS = 50

CF_DIMS = 300
CF_EPOCHS = 50
CF_MAX_SAMPLED = 40
CF_LR = 0.05
TRAIN_FILE = "/data1/playlists/train.json"

STRATEGY = 'strat_gnr_cf'

# write to
SAVE_DATASET_LOCATION = '/data1/playlists/'
DATASET_NAME_TAGS = 'spec_tags_{}'.format(STRATEGY)
#ID2TOKEN_NAME = '../json/id2token_{}.json'.format(STRATEGY)
#SCALER_NAME = '../scaler_{}.pkl'.format(STRATEGY)

# read from
SOUND_TAGS = '../tags/track_tags_gnr.json'
#SOUND_TAGS = '../tags/track_tags_gnr_v3.json'
SPECTROGRAM_LOCATION = '/data1/playlists/npy'  # SET FOLDER HERE!


def train_mf(impl_train_data, dims=200, epochs=50, max_sampled=10, lr=0.05):

    model = LightFM(loss='warp', no_components=dims, max_sampled=max_sampled, learning_rate=lr, random_state=42)
    model = model.fit(impl_train_data, epochs=epochs, num_threads=24)

    user_biases, user_embeddings = model.get_user_representations()
    item_biases, item_embeddings = model.get_item_representations()
    item_vec = np.concatenate((item_embeddings, np.reshape(item_biases, (1, -1)).T), axis=1)
    user_vec = np.concatenate((user_embeddings, np.ones((1, user_biases.shape[0])).T), axis=1)

    return user_vec, item_vec

def load_cf_data(train_file, tracks_ids):
    train_playlists = json.load(open(train_file, encoding='utf-8'))

    rows= []
    cols= []
    data= []
    playlists_ids = []
    playlists_test = {}
    for playlist in train_playlists:
        for track in playlist['songs']:
            if str(track) in tracks_ids:
                cols.append(tracks_ids[str(track)])
                rows.append(len(playlists_ids))
                data.append(1)
            else:
                if str(playlist['id']) not in playlists_test:
                    playlists_test[str(playlist['id'])] = []
                playlists_test[str(playlist['id'])].append(str(track))
        playlists_ids.append(playlist['id'])
    train_coo = sparse.coo_matrix((data, (rows, cols)), dtype=np.float32)
    json.dump(playlists_test, open(os.path.join(SAVE_DATASET_LOCATION, 'test_playlists.json'), 'w'))
    json.dump(playlists_ids, open(os.path.join(SAVE_DATASET_LOCATION, 'all_playlists_ids.json'), 'w'))
    user_vec, item_vec = train_mf(train_coo, dims=CF_DIMS, epochs=CF_EPOCHS, max_sampled=CF_MAX_SAMPLED, lr=CF_LR)
    np.save(open(os.path.join(SAVE_DATASET_LOCATION, 'playlists_cf_vec.npy'), 'wb'), user_vec)

    return item_vec


if __name__ == "__main__":
    # load sound tags and create label idx vectors
    sound_tags = json.load(open(SOUND_TAGS))
    num_sounds = len(sound_tags)
    all_tags = {}
    for tags_list in sound_tags.values():
        for t in tags_list:
            if t not in all_tags:
                all_tags[t] = 1
    print('\n Finished loading tags\n')

    sound_tags_num = {i:k for i,k in enumerate(sound_tags.keys())}

    mlb = MultiLabelBinarizer()
    binarized_gnr = mlb.fit_transform(list(sound_tags.values()))
    print("Dimensions", binarized_gnr.shape)
    msss = MultilabelStratifiedShuffleSplit(test_size=0.1, random_state=0, n_splits=1)
    r = list(msss.split(list(sound_tags_num.keys()), binarized_gnr))
    train_ids_tmp = [sound_tags_num[i] for i in r[0][0]]
    test_ids = [sound_tags_num[i] for i in r[0][1]]

    sound_tags_num = {i:k for i,k in enumerate(train_ids_tmp)}
    inv_sound_tags_num = {v:k for k,v in sound_tags_num.items()}

    # train MF model
    item_vec = load_cf_data(TRAIN_FILE, inv_sound_tags_num)

    binarized_gnr = mlb.fit_transform([sound_tags[k] for k in train_ids_tmp])
    msss = MultilabelStratifiedShuffleSplit(test_size=0.1, random_state=0, n_splits=1)
    r = list(msss.split(list(sound_tags_num.keys()), binarized_gnr))
    train_ids = [sound_tags_num[i] for i in r[0][0]]
    val_ids = [sound_tags_num[i] for i in r[0][1]]

    # Divide CF vectors for train and val
    train_cf_vec = item_vec[r[0][0]]
    val_cf_vec = item_vec[r[0][1]]


    num_training_instances = len(train_ids)
    num_validation_instances = len(val_ids)
    num_test_instances = len(test_ids)

    print('Num training instances: {}'.format(num_training_instances))
    print('Num validation instances: {}'.format(num_validation_instances))


    # tag label dataset
    hdf5_file_tags = h5py.File('{}/{}'.format(SAVE_DATASET_LOCATION, DATASET_NAME_TAGS), mode='w')
    ds_group = hdf5_file_tags.create_group('dataset')
    ds_group.create_dataset("id", (num_training_instances, 1), dtype='int32')
    ds_group.create_dataset("data", (num_training_instances, NUM_FRAMES, NUM_BANDS), dtype='float32')
    ds_group.create_dataset("cf_data", (num_training_instances, CF_DIMS+1), dtype='float32')
    ds_group.create_dataset("label", (num_training_instances, NUM_TAGS), dtype='int16')

    hdf5_file_tags_val = h5py.File('{}/{}_val'.format(SAVE_DATASET_LOCATION, DATASET_NAME_TAGS), mode='w')
    ds_group_val = hdf5_file_tags_val.create_group('dataset')
    ds_group_val.create_dataset("id", (num_validation_instances, 1), dtype='int32')
    ds_group_val.create_dataset("data", (num_validation_instances, NUM_FRAMES, NUM_BANDS), dtype='float32')
    ds_group_val.create_dataset("cf_data", (num_training_instances, CF_DIMS+1), dtype='float32')
    ds_group_val.create_dataset("label", (num_validation_instances, NUM_TAGS), dtype='int16')

    hdf5_file_tags_test = h5py.File('{}/{}_test'.format(SAVE_DATASET_LOCATION, DATASET_NAME_TAGS), mode='w')
    ds_group_test = hdf5_file_tags_test.create_group('dataset')
    ds_group_test.create_dataset("id", (num_test_instances, 1), dtype='int32')
    ds_group_test.create_dataset("data", (num_test_instances, NUM_FRAMES, NUM_BANDS), dtype='float32')
    ds_group_test.create_dataset("label", (num_test_instances, NUM_TAGS), dtype='int16')


    progress_bar = ProgressBar(len(train_ids), 20, 'Dataset train tags')
    progress_bar.update(0)
    count_chunks = 0

    for idx, sound_id in enumerate(train_ids):
        try:
            sound_str = str(sound_id)
            spec_filename = '{}/{}.npy'.format(sound_str[:-3], sound_id)
            if len(sound_str)<=3:
                spec_filename = '{}/{}.npy'.format('0', sound_id)

            x = np.load(os.path.join(SPECTROGRAM_LOCATION, spec_filename))
            progress_bar.update(idx)
            if x.any():
                ds_group["id"][count_chunks] = int(sound_id)
                ds_group["data"][count_chunks] = x.T[:NUM_FRAMES,:]
                ds_group["cf_data"][count_chunks] = train_cf_vec[idx]
                #ds_group["label"][count_chunks] = np.array(sound_tags[sound_id])
                ds_group["label"][count_chunks] = np.array(sound_tags[sound_id]+[-1]*(NUM_TAGS-len(sound_tags[sound_id])))
                count_chunks += 1
        except Exception as e:
            print(e)
            pass

    print('\n Train Tags Dataset finished, created {} training instances from {} audio files'.format(count_chunks, len(train_ids)))

    progress_bar = ProgressBar(len(val_ids), 20, 'Dataset val tags')
    progress_bar.update(0)
    count_chunks = 0

    for idx, sound_id in enumerate(val_ids):
        try:
            progress_bar.update(idx)
            sound_str = str(sound_id)
            spec_filename = '{}/{}.npy'.format(sound_str[:-3], sound_id)
            if len(sound_str)<=3:
                spec_filename = '{}/{}.npy'.format('0', sound_id)

            x = np.load(os.path.join(SPECTROGRAM_LOCATION, spec_filename))
            if x.any():
                ds_group_val["id"][count_chunks] = int(sound_id)
                ds_group_val["data"][count_chunks] = x.T[:NUM_FRAMES,:]
                ds_group_val["cf_data"][count_chunks] = val_cf_vec[idx]
                #ds_group_val["label"][count_chunks] = np.array(sound_tags[sound_id])
                ds_group_val["label"][count_chunks] = np.array(sound_tags[sound_id]+[-1]*(NUM_TAGS-len(sound_tags[sound_id])))

                count_chunks += 1

        except Exception as e:
            print(e)
            pass

    print('\n Val Tags Dataset finished, created {} training instances from {} audio files\n'.format(count_chunks, len(val_ids)))

    progress_bar = ProgressBar(len(test_ids), 20, 'Dataset test tags')
    progress_bar.update(0)
    count_chunks = 0

    for idx, fs_id in enumerate(test_ids):
        try:
            progress_bar.update(idx)
            sound_str = str(sound_id)
            spec_filename = '{}/{}.npy'.format(sound_str[:-3], sound_id)
            if len(sound_str)<=3:
                spec_filename = '{}/{}.npy'.format('0', sound_id)

            x = np.load(os.path.join(SPECTROGRAM_LOCATION, spec_filename))
            if x.any():
                ds_group_test["id"][count_chunks] = int(sound_id)
                ds_group_test["data"][count_chunks] = x.T[:NUM_FRAMES,:]
                ds_group_test["label"][count_chunks] = np.array(sound_tags[sound_id]+[-1]*(NUM_TAGS-len(sound_tags[sound_id])))

                count_chunks += 1

        except Exception as e:
            print(e)
            pass

    hdf5_file_tags.close()
    hdf5_file_tags_val.close()
