import copy
import logging
from loguru import logger
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import save_config_file

'''
refer: https://github.com/vdogmcgee/SimCSE-Chinese-Pytorch/blob/main/simcse_unsup.py
'''

class SimCSE(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.tokenizer = kwargs['tokenizer']

        if self.args.do_train:
            os.makedirs(self.args.output_path, exist_ok=True)
            log_dir = self.args.output_path
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

            logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        else:
            pass

    # bsz : batch size (number of positive pairs)
    # d   : latent dim
    # x   : Tensor, shape=[bsz, d]
    #       latents for one side of positive pairs
    # y   : Tensor, shape=[bsz, d]
    #       latents for the other side of positive pairs
    # Thanks to https://github.com/SsnL/align_uniform

    def simcse_unsup_loss(self, y_pred: 'tensor') -> 'tensor':
        """
        loss function for self-supervised training
        y_pred (tensor): bert output, [batch_size * 2, 768]

        """
        # Get the label for each prediction, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        y_true = torch.arange(y_pred.shape[0], device=self.args.device)
        labels = (y_true - y_true % 2 * 2) + 1
        # Calculate the similarity matrix
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # Discard the diagnosis (we don't want the self-similarity)
        sim = sim - torch.eye(y_pred.shape[0], device=self.args.device) * 1e12
        # similarity divide by temperature
        logits = sim / self.args.temperature

        return logits, labels

    def get_perm(self, x):
        """get random permutation"""
        size = x.size()[0]
        if self.args.device == "cuda":
            index = torch.randperm(size).cuda()
        else:
            index = torch.randperm(size)
        return index

    def tokenize(self, data, step):
        anchor = data["anchor"]
        positive = data["positive"]

        if step % self.args.aug_every_n_steps == 0:
            if self.args.do_tf_idf and self.args.do_hard_negatives:
                anchor_hn = data["anchor_hn"]
                positive_hn = data["positive_hn"]
                sample = [self.tokenizer([anchor[i], positive[i], anchor_hn[i], positive_hn[i]],
                                            max_length=self.args.max_len,
                                            truncation=True,
                                            padding="max_length",
                                            return_tensors='pt') for i in range(len(anchor))]
                return sample

        sample = [self.tokenizer([anchor[i], positive[i]],
                                 max_length=self.args.max_len,
                                 truncation=True,
                                 padding="max_length",
                                 return_tensors='pt') for i in range(len(anchor))]
        return sample


    def train(self, train_loader, eval_loader):
        logger.info("start training")
        self.model.train()
        device = self.args.device
        best = 0
        best_step = 0
        total_steps = len(train_loader)*self.args.epochs

        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start SimCSE training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch in range(self.args.epochs):
            pbar = tqdm(train_loader)
            for batch_idx, sample in enumerate(pbar):
                step = epoch * len(train_loader) + batch_idx
                # [batch, n, seq_len] -> [batch * n, sql_len]
                data = self.tokenize(sample, step)

                input_ids = torch.cat([data[i]['input_ids'] for i in range(len(data))]).to(device)
                attention_mask = torch.cat([data[i]['attention_mask'] for i in range(len(data))]).to(device)
                token_type_ids = torch.cat([data[i]['token_type_ids'] for i in range(len(data))]).to(device)
                out = self.model(input_ids, attention_mask, token_type_ids)

                logits, labels = self.simcse_unsup_loss(out)

                loss = F.cross_entropy(logits, labels)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                step += 1
                self.scheduler.step()

                pbar.set_description(
                    "Progress: {:.2%} | Loss: {:.3} | lr: {:.3}".format(step / total_steps, loss, self.scheduler.get_last_lr()[0]))

                if (step) % self.args.log_every_n_steps == 0:
                    corrcoef = self.evaluate(model=self.model, dataloader=eval_loader)
                    logger.info('loss:{:.3}, corrcoef: {:.3} in step {} epoch {}'.format(loss, corrcoef, step, epoch + 1))
                    logging.debug(
                        "Progress:{:.2%}\tEpoch: {}\tStep: {}\tlr: {:.3}\tLoss: {:.4}\tcorrcoef: {:.4}\tBest score {:.4} at step: {}".format(
                            step / total_steps, epoch, step, self.scheduler.get_last_lr()[0], loss, corrcoef, float(best), best_step))
                    self.writer.add_scalar('loss', loss, step)
                    self.writer.add_scalar('corrcoef', corrcoef, step)
                    self.model.train()

                    # Save the model if the evalaute result outperform the previous
                    if best < corrcoef:
                        best = corrcoef
                        best_step = step
                        if self.args.save_data:
                            torch.save(self.model.state_dict(), os.path.join(self.args.output_path, 'simcse.pt'))
                            checkpoint_name = 'checkpoint_best'
                            try:
                                self.model.model.save_pretrained(os.path.join(self.writer.log_dir, checkpoint_name))
                            except:
                                self.model.module.model.save_pretrained(os.path.join(self.writer.log_dir, checkpoint_name))
                        logger.info('higher corrcoef: {}'.format(best))

        logging.debug(f"Best corrcoef {best} at step: {best_step}")

        if self.args.save_data:
            torch.save(self.model.state_dict(), os.path.join(self.args.output_path, 'final_simcse.pt'))
            checkpoint_name = 'checkpoint_final'
            try:
                self.model.model.save_pretrained(os.path.join(self.writer.log_dir, checkpoint_name))
            except:
                self.model.module.model.save_pretrained(os.path.join(self.writer.log_dir, checkpoint_name))


    def evaluate(self, model, dataloader):
        model.eval()
        sim_tensor = torch.tensor([], device=self.args.device)
        label_array = np.array([])
        with torch.no_grad():
            for source, target, label in dataloader:
                # source        [batch, 1, seq_len] -> [batch, seq_len]
                source_input_ids = source.get('input_ids').squeeze(1).to(self.args.device)
                source_attention_mask = source.get('attention_mask').squeeze(1).to(self.args.device)
                if self.args.arch == 'roberta':
                    source_pred = model(source_input_ids, source_attention_mask, is_train=False)
                else:
                    source_token_type_ids = source.get('token_type_ids').squeeze(1).to(self.args.device)
                    source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids, is_train=False)

                # target        [batch, 1, seq_len] -> [batch, seq_len]
                target_input_ids = target.get('input_ids').squeeze(1).to(self.args.device)
                target_attention_mask = target.get('attention_mask').squeeze(1).to(self.args.device)
                if self.args.arch == 'roberta':
                    target_pred = model(source_input_ids, source_attention_mask, is_train=False)
                else:
                    target_token_type_ids = target.get('token_type_ids').squeeze(1).to(self.args.device)
                    target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids, is_train=False)

                # concat
                sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
                sim_tensor = torch.cat((sim_tensor, sim), dim=0)
                label_array = np.append(label_array, np.array(label))
        # corrcoef
        return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation