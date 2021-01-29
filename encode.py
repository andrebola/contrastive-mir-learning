"""
This script is used to compute neural network embeddings.
"""
import torch
import numpy as np
import sklearn
import pickle
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import librosa

from utils import extract_spectrogram
from models import AudioEncoder


def return_loaded_model(Model, checkpoint):
    model = Model()
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.eval()
    return model


def extract_audio_embedding(model, filename):
    with torch.no_grad():
        try:
            x = extract_spectrogram(filename)
            x = scaler.transform(x)
            x = torch.unsqueeze(torch.unsqueeze(torch.tensor(x), 0), 0).float()
            embedding, embedding_d = model(x)
            return embedding, embedding_d
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e, filename)


def extract_audio_embedding_chunks(model, filename):
    with torch.no_grad():
        try:
            x = extract_spectrogram(filename)
            x_chunks = librosa.util.frame(np.asfortranarray(x), frame_length=256, hop_length=256, axis=-1)
            x_chunks = torch.tensor(x_chunks).permute(2, 0, 1)
            x_chunks = torch.unsqueeze(x_chunks, 1)
            embedding_chunks, embedding_d_chunks = model(x_chunks)
            return embedding_chunks, embedding_d_chunks
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e, filename)


if __name__ == "__main__":
    for MODEL_NAME in [
        'minz_att_4h_w2v_128/audio_encoder_epoch_120',
    ]:
        MODEL_PATH = f'./saved_models/{MODEL_NAME}.pt'

        model = return_loaded_model(AudioEncoder, MODEL_PATH)
        model.eval()

        # GTZAN
        p = Path('./data/GTZAN/genres')
        filenames_gtzan = p.glob('**/*.wav')

        # # US8K
        # p = Path('./data/UrbanSound8K/audio')
        # filenames_us8k = p.glob('**/*.wav')
        # 
        # # NSynth
        # p = Path('./data/nsynth/nsynth-train/audio_selected')
        # filenames_nsynth_train = p.glob('*.wav')
        # p = Path('./data/nsynth/nsynth-test/audio')
        # filenames_nsynth_test = p.glob('*.wav')
        # 
        # dataset_files = [filenames_gtzan, filenames_us8k, filenames_nsynth_train, filenames_nsynth_test]
        # dataset_names = ['gtzan', 'us8k', 'nsynth/train', 'nsynth/test']

        dataset_files = [filenames_gtzan] 
        dataset_names = ['gtzan']

        for filenames, ds_name in zip(dataset_files, dataset_names):

            print(f'\n {ds_name}  {MODEL_NAME}')

            for f in tqdm(filenames):
                try:
                    with torch.no_grad():
                        model_name = MODEL_NAME.split('/')[0] + '_' + MODEL_NAME.split('_epoch_')[-1]
                        folder = f'./data/embeddings/{ds_name}/embeddings_{model_name}'
                        Path(folder).mkdir(parents=True, exist_ok=True)
                        embedding, embedding_d = extract_audio_embedding_chunks(model, str(f))
                        np.save(Path(folder, str(f.stem)+'.npy'), embedding)
                except Exception as e:
                    print(e)
            print('\n')
