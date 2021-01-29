"""
This file regroups the procedures for training the neural networks.
A training uses a configuration json file (e.g. configs/dual_ae_c.json).
"""
from pathlib import Path
from itertools import chain
import torch
from torch.utils import data
from torch import nn, optim
import math
#from torchvision.utils import save_image
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
import json

from data_loader import InMemoryDataset
from models import *
from utils import kullback_leibler, contrastive_loss


class Trainer():
    def __init__(self, params):
        self.params = params
        self.audio_encoder = None
        self.tag_encoder = None
        self.train_dataset_file = params['train_dataset_file']
        self.validation_dataset_file = params['validation_dataset_file']
        self.contrastive_temperature = params['contrastive_temperature']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.device = torch.device(params['device'])
        self.experiment_name = params['experiment_name']
        self.log_interval = params['log_interval']
        self.save_model_every = params['save_model_every']
        self.max_num_tags = 10   # This may be increased and used in data_loader.py
        # This can be set here, or maybe automaticaly read from embedding_matrix_128.npy
        # don't forget to add +1 to the size of voc for adding idx 0 for no tag.
        self.w2v_emb_file = 'embedding_matrix_128.npy'
        w2v_embeddings = np.load(self.w2v_emb_file)
        # We add one (+1) so we also consider no tag
        self.size_voc = w2v_embeddings.shape[0] + 1
        self.aggregation = params['aggregation']
        self.num_heads = params['num_heads']
        self.encoder_type = params['encoder']
        self.save_model_loc = '/data1/playlists/models/contrastive'
        self.curr_min_val = 10

    def init_models(self):
        self.audio_encoder = AudioEncoder(128)
        if self.aggregation == 'mean':
            self.tag_encoder = TagMeanEncoder(self.size_voc, 128, 128, emb_file=self.w2v_emb_file)
        elif self.aggregation == 'self':
            self.tag_encoder = TagSelfAttentionEncoder(self.max_num_tags, 128, self.num_heads, 128, 128, 128, self.w2v_emb_file,dropout=0.1)
        self.cf_encoder = CFEncoder(301, 128)

    def load_model_checkpoints(self):
        saved_models_folder = Path(self.save_model_loc, self.experiment_name)
        try:
            last_epoch = max([int(f.stem.split('epoch_')[-1]) for f in saved_models_folder.iterdir()])
            self.audio_encoder.load_state_dict(torch.load(str(Path(self.save_model_loc, self.experiment_name, f'audio_encoder_epoch_{last_epoch}.pt'))))
            self.tag_encoder.load_state_dict(torch.load(str(Path(self.save_model_loc, self.experiment_name, f'tag_encoder_att_epoch_{last_epoch}.pt'))))
            print(f'Model checkpoints from epoch {last_epoch} loaded...')
        except ValueError:
            last_epoch = 0
            print('No model loaded, training from scratch...')

        self.iteration_idx = last_epoch * int(self.length_val_dataset / self.batch_size)
        self.last_epoch = last_epoch

    def train(self):
        """ Train models

        """
        # Data loaders
        loader_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': 2,
            'drop_last': True,
        }

        dataset_train = InMemoryDataset(self.train_dataset_file)
        #dataset_train = InMemoryDataset(self.validation_dataset_file)
        dataset_val = InMemoryDataset(self.validation_dataset_file)

        self.train_loader = data.DataLoader(dataset_train, **loader_params)
        self.val_loader = data.DataLoader(dataset_val, **loader_params)
        self.length_train_dataset = len(self.train_loader.dataset)
        self.length_val_dataset = len(self.val_loader.dataset)

        # folder for model checkpoints
        model_checkpoints_folder = Path(self.save_model_loc, self.experiment_name)
        if not model_checkpoints_folder.exists():
            model_checkpoints_folder.mkdir()

        # models
        self.init_models()
        self.load_model_checkpoints()

        self.audio_encoder.to(self.device)
        if self.encoder_type == "gnr":
            self.tag_encoder.to(self.device)
        if self.encoder_type == "MF":
            self.cf_encoder.to(self.device)

        # optimizers
        if self.encoder_type == "gnr":
            self.opt = torch.optim.Adam(chain(self.audio_encoder.parameters(), self.tag_encoder.parameters()), lr=self.learning_rate, weight_decay=1e-4)
        if self.encoder_type == "MF":
            self.opt = torch.optim.Adam(chain(self.audio_encoder.parameters(), self.cf_encoder.parameters()), lr=self.learning_rate, weight_decay=1e-4)

        # tensorboard
        with SummaryWriter(log_dir=str(Path('runs', self.experiment_name)), max_queue=100) as self.tb:

            # Training loop
            for epoch in range(self.last_epoch+1, self.epochs + 1):
                self.train_one_epoch(epoch)
                self.val(epoch)

    def train_one_epoch(self, epoch):
        """ Train one epoch

        """
        self.audio_encoder.train()
        if self.encoder_type == "gnr":
            self.tag_encoder.train()
        if self.encoder_type == "MF":
            self.cf_encoder.train()

        # losses
        train_pairwise_loss = 0

        for batch_idx, (data, tags, cf_embeddings, sound_ids) in enumerate(self.train_loader):
            self.iteration_idx += 1

            # TODO: REMOVE THAT
            # tags should already in the tag_idxs form, except for the +1 to indexes to use idx 0 for no tag
            # We probably want to add some pre-processing in data_loader.py
            # e.g. select random tags from the 100, or select random sepctrogram chunk
            """
            tag_idxs = [
                ([idx+1 for idx, val in enumerate(tag_v) if val]
                 + self.max_num_tags*[0])[:self.max_num_tags]
                for tag_v in tags
            ]

            """
            # encode
            x = data.view(-1, 1, 48, 256).to(self.device)
            z_audio, z_d_audio = self.audio_encoder(x)

            if self.encoder_type == "gnr":
                curr_labels = []
                for curr_tags in tags:
                    non_neg = [i+1 for i in curr_tags if i != -1]
                    new_tags = np.zeros(self.max_num_tags)
                    #new_tags[:len(non_neg)] = np.random.choice(non_neg, min(self.max_num_tags, len(non_neg)), replace=False)
                    new_tags[:min(len(non_neg), 10)] = non_neg[:10]
                    curr_labels.append(new_tags)
                tags_input = torch.tensor(curr_labels, dtype=torch.long).to(self.device)
                #tags_input = tags.to(self.device)
                z_tags, attn = self.tag_encoder(tags_input, z_d_audio, mask=tags_input.unsqueeze(1))
                pairwise_loss = contrastive_loss(z_d_audio, z_tags, self.contrastive_temperature)
            if self.encoder_type == "MF" :
                cf_input = cf_embeddings.to(self.device)
                z_cf = self.cf_encoder(cf_input)

                # contrastive loss
                pairwise_loss = contrastive_loss(z_d_audio, z_cf, self.contrastive_temperature)

            # Optimize models
            self.opt.zero_grad()
            pairwise_loss.backward()
            self.opt.step()


            train_pairwise_loss += pairwise_loss.item()

            # write to tensorboard
            # These are too many data to send to tensorboard, but it can be useful for debugging/developing
            if False:
                self.tb.add_scalar("iter/contrastive_pairwise_loss", pairwise_loss.item(), self.iteration_idx)

            # logs per batch
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPairwise loss: {:.4f})'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        pairwise_loss.item()
                    )
                )

        # epoch logs
        train_pairwise_loss = train_pairwise_loss / self.length_train_dataset * self.batch_size
        print('====> Epoch: {}  Pairwise loss: {:.8f}'.format(epoch, train_pairwise_loss))
        print('\n')

        # tensorboard
        self.tb.add_scalar("contrastive_pairwise_loss/train/sum", train_pairwise_loss, epoch)

        if epoch%self.save_model_every == 0:
            torch.save(self.audio_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'audio_encoder_epoch_{epoch}.pt')))
            if self.encoder_type == "gnr":
                torch.save(self.tag_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'tag_encoder_att_epoch_{epoch}.pt')))
            if self.encoder_type == "MF":
                torch.save(self.cf_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'cf_encoder_att_epoch_{epoch}.pt')))

    def val(self, epoch):
        """ Validation

        """
        # A little bit a code repeat here...
        self.audio_encoder.eval()
        if self.encoder_type == "gnr":
            self.tag_encoder.eval()
        if self.encoder_type == "MF":
            self.cf_encoder.eval()

        val_pairwise_loss = 0

        with torch.no_grad():
            for i, (data, tags, cf_embeddings, sound_ids) in enumerate(self.val_loader):
                """
                curr_labels = []
                for curr_tags in tags:
                    non_neg = [i+1 for i in curr_tags if i != -1]
                    new_tags = np.zeros(self.max_num_tags)
                    #new_tags[:len(non_neg)] = np.random.choice(non_neg, min(self.max_num_tags, len(non_neg)), replace=False)
                    new_tags[:min(len(non_neg), 10)] = non_neg[:10]
                    curr_labels.append(new_tags)
                tags_input = torch.tensor(curr_labels, dtype=torch.long).to(self.device)
                """

                x = data.view(-1, 1, 48, 256).to(self.device)
                cf_input = cf_embeddings.to(self.device)

                # encode
                z_audio, z_d_audio = self.audio_encoder(x)
                if self.encoder_type == "gnr" :
                    z_tags, attn = self.tag_encoder(tags_input, z_d_audio, mask=tags_input.unsqueeze(1))
                    # pairwise correspondence loss
                    pairwise_loss = contrastive_loss(z_d_audio, z_tags, self.contrastive_temperature)
                if self.encoder_type == "MF" :
                    z_cf = self.cf_encoder(cf_input)
                    # pairwise correspondence loss
                    pairwise_loss = contrastive_loss(z_d_audio, z_cf, self.contrastive_temperature)

                val_pairwise_loss += pairwise_loss.item()

        val_pairwise_loss = val_pairwise_loss / self.length_val_dataset * self.batch_size

        print('====> Val average pairwise loss: {:.4f}'.format(val_pairwise_loss))
        print('\n\n')

        # tensorboard
        self.tb.add_scalar("contrastive_pairwise_loss/val/sum", val_pairwise_loss, epoch)
        if not (math.isinf(val_pairwise_loss) or math.isinf(val_pairwise_loss)):
            if val_pairwise_loss<self.curr_min_val:
                self.curr_min_val = val_pairwise_loss
                torch.save(self.audio_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'audio_encoder_epoch_best.pt')))
                if self.encoder_type == "gnr":
                    torch.save(self.tag_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'tag_encoder_att_epoch_best.pt')))
                if self.encoder_type == "MF":
                    torch.save(self.cf_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'cf_encoder_att_epoch_best.pt')))


