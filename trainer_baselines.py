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
        self.mo_encoder = Multiobjective(301,self.size_voc)

    def load_model_checkpoints(self):
        saved_models_folder = Path(self.save_model_loc, self.experiment_name)
        try:
            last_epoch = max([int(f.stem.split('epoch_')[-1]) for f in saved_models_folder.iterdir()])
            self.audio_encoder.load_state_dict(torch.load(str(Path(self.save_model_loc, self.experiment_name, f'audio_encoder_epoch_{last_epoch}.pt'))))
            self.mo_encoder.load_state_dict(torch.load(str(Path(self.save_model_loc, self.experiment_name, f'mo_encoder_epoch_{last_epoch}.pt'))))
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
        self.mo_encoder.to(self.device)

        # optimizers
        self.opt = torch.optim.Adam(chain(self.audio_encoder.parameters(), self.mo_encoder.parameters()), lr=self.learning_rate, weight_decay=1e-4)

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
        self.mo_encoder.train()

        # losses
        train_pairwise_loss = 0
        train_pairwise_loss_1 = 0
        train_pairwise_loss_2 = 0
        train_pairwise_loss_3 = 0

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

            x = data.view(-1, 1, 48, 256).to(self.device)

            if self.encoder_type == "gnr" or self.encoder_type == "MF_gnr":
                curr_labels = []
                for curr_tags in tags:
                    non_neg = [i for i in curr_tags if i != -1]
                    new_tags = np.zeros(self.size_voc)
                    new_tags[non_neg] = 1
                    curr_labels.append(new_tags)
                tags_input = torch.tensor(curr_labels, dtype=torch.float).to(self.device)

            if self.encoder_type == "MF" or self.encoder_type == "MF_gnr":
                cf_input = cf_embeddings.to(self.device)

            # encode
            z_audio, z_d_audio = self.audio_encoder(x)
            z_cf, z_tags = self.mo_encoder(z_d_audio)

            # contrastive loss
            if self.encoder_type == "gnr" or self.encoder_type == "MF_gnr":
                bceloss = torch.nn.BCEWithLogitsLoss()
                pairwise_loss_1 = bceloss(z_tags, tags_input)
                pairwise_loss = pairwise_loss_1
            if self.encoder_type == "MF" or self.encoder_type == "MF_gnr":
                mseloss = torch.nn.MSELoss()
                pairwise_loss_2 = mseloss(z_cf, cf_input)
                pairwise_loss = pairwise_loss_2
            if self.encoder_type == "MF_gnr":
                pairwise_loss = pairwise_loss_1 + pairwise_loss_2

            # Optimize models
            self.opt.zero_grad()
            pairwise_loss.backward()
            self.opt.step()


            train_pairwise_loss += pairwise_loss.item()
            if self.encoder_type == "gnr":
                train_pairwise_loss_1 += pairwise_loss_1.item()
            if self.encoder_type == "MF":
                train_pairwise_loss_2 += pairwise_loss_2.item()

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
        if self.encoder_type == "gnr":
            train_pairwise_loss_1 = train_pairwise_loss_1 / self.length_train_dataset * self.batch_size
        if self.encoder_type == "MF":
            train_pairwise_loss_2 = train_pairwise_loss_2 / self.length_train_dataset * self.batch_size
        print('====> Epoch: {}  Pairwise loss: {:.8f}'.format(epoch, train_pairwise_loss))
        print('\n')

        # tensorboard
        self.tb.add_scalar("contrastive_pairwise_loss/train/sum", train_pairwise_loss, epoch)
        if self.encoder_type == "gnr":
            self.tb.add_scalar("contrastive_pairwise_loss/train/1", train_pairwise_loss_1, epoch)
        if self.encoder_type == "MF":
            self.tb.add_scalar("contrastive_pairwise_loss/train/2", train_pairwise_loss_2, epoch)

        if epoch%self.save_model_every == 0:
            torch.save(self.audio_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'audio_encoder_epoch_{epoch}.pt')))
            torch.save(self.mo_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'mo_encoder_epoch_{epoch}.pt')))

    def val(self, epoch):
        """ Validation

        """
        # A little bit a code repeat here...
        self.audio_encoder.eval()
        self.mo_encoder.eval()

        val_pairwise_loss = 0
        val_pairwise_loss_1 = 0
        val_pairwise_loss_2 = 0
        val_pairwise_loss_3 = 0

        with torch.no_grad():
            for i, (data, tags, cf_embeddings, sound_ids) in enumerate(self.val_loader):

                x = data.view(-1, 1, 48, 256).to(self.device)

                if self.encoder_type == "gnr" or self.encoder_type == "MF_gnr":
                    curr_labels = []
                    for curr_tags in tags:
                        non_neg = [i for i in curr_tags if i != -1]
                        new_tags = np.zeros(self.size_voc)
                        new_tags[non_neg] = 1
                        curr_labels.append(new_tags)
                    tags_input = torch.tensor(curr_labels, dtype=torch.float).to(self.device)

                if self.encoder_type == "MF" or self.encoder_type == "MF_gnr":
                    cf_input = cf_embeddings.to(self.device)

                # encode
                z_audio, z_d_audio = self.audio_encoder(x)
                z_cf, z_tags = self.mo_encoder(z_d_audio)

                # contrastive loss
                if self.encoder_type == "gnr" or self.encoder_type == "MF_gnr":
                    bceloss = torch.nn.BCEWithLogitsLoss()
                    pairwise_loss_1 = bceloss(z_tags, tags_input)
                    pairwise_loss = pairwise_loss_1
                if self.encoder_type == "MF" or self.encoder_type == "MF_gnr":
                    mseloss = torch.nn.MSELoss()
                    pairwise_loss_2 = mseloss(z_cf, cf_input)
                    pairwise_loss = pairwise_loss_2
                if self.encoder_type == "MF_gnr":
                    pairwise_loss = pairwise_loss_1 + pairwise_loss_2


                val_pairwise_loss += pairwise_loss.item()
                if self.encoder_type == "gnr":
                    val_pairwise_loss_1 += pairwise_loss_1.item()
                if self.encoder_type == "MF":
                    val_pairwise_loss_2 += pairwise_loss_2.item()

        val_pairwise_loss = val_pairwise_loss / self.length_val_dataset * self.batch_size
        if self.encoder_type == "gnr":
            val_pairwise_loss_1 = val_pairwise_loss_1 / self.length_val_dataset * self.batch_size
        if self.encoder_type == "MF":
            val_pairwise_loss_2 = val_pairwise_loss_2 / self.length_val_dataset * self.batch_size

        print('====> Val average pairwise loss: {:.4f}'.format(val_pairwise_loss))
        print('\n\n')

        # tensorboard
        self.tb.add_scalar("contrastive_pairwise_loss/val/sum", val_pairwise_loss, epoch)
        if self.encoder_type == "gnr":
            self.tb.add_scalar("contrastive_pairwise_loss/val/1", val_pairwise_loss_1, epoch)
        if self.encoder_type == "MF":
            self.tb.add_scalar("contrastive_pairwise_loss/val/2", val_pairwise_loss_2, epoch)
        if not (math.isinf(val_pairwise_loss) or math.isinf(val_pairwise_loss)):
            if val_pairwise_loss<self.curr_min_val:
                self.curr_min_val = val_pairwise_loss
                torch.save(self.audio_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'audio_encoder_epoch_best.pt')))
                torch.save(self.mo_encoder.state_dict(), str(Path(self.save_model_loc, self.experiment_name, f'mo_encoder_epoch_best.pt')))


