import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel, RobertaConfig, RobertaModel, ElectraConfig, ElectraModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
# from transformers.models.electra.modeling_electra import


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.args = kwargs['args']
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if self.args.activation == "relu":
            self.activation = nn.ReLU(inplace=False)
            # self.dropout = nn.Dropout(self.args.dropout)
        elif self.args.activation == "tanh":
            self.activation = nn.Tanh()
            # self.dropout = nn.Dropout(self.args.dropout)
        else:
            raise ValueError

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class SimcseModelUnsup(nn.Module):
    """Simcse unsupercised"""

    def __init__(self, *args, **kwargs):
        super(SimcseModelUnsup, self).__init__()
        self.args = kwargs['args']

        pretrained_model = self.args.pretrained_model
        if pretrained_model is None:
            config = BertConfig()
            self.model = BertModel(config)
        elif self.args.arch == 'roberta':
            config = RobertaConfig.from_pretrained(pretrained_model)
            config.attention_probs_dropout_prob = self.args.dropout
            config.hidden_dropout_prob = self.args.dropout
            self.model = AutoModel.from_pretrained(pretrained_model, config=config)
            if self.args.do_mlm:
                self.lm_head = RobertaLMHead(config)
        elif self.args.arch == 'bert':
            config = BertConfig.from_pretrained(pretrained_model)
            config.attention_probs_dropout_prob = self.args.dropout
            config.hidden_dropout_prob = self.args.dropout
            self.model = AutoModel.from_pretrained(pretrained_model, config=config)
            if self.args.do_mlm:
                self.lm_head = BertLMPredictionHead(config)
        elif self.args.arch == 'electra':
            config = ElectraConfig.from_pretrained(pretrained_model)
            config.attention_probs_dropout_prob = self.args.dropout
            config.hidden_dropout_prob = self.args.dropout
            self.model = AutoModel.from_pretrained(pretrained_model, config=config)
            # if self.do_mlm:
        else:
            raise ValueError("Unsupported pretrained model")

        self.mlp = MLPLayer(config, args=self.args)

    def forward(self, input_ids, attention_mask, token_type_ids=None, output_hidden_states=True, is_train=True):
        if self.args.arch == 'roberta':
            out = self.model(input_ids, attention_mask, output_hidden_states=output_hidden_states, return_dict=True)
        else:
            out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=output_hidden_states, return_dict=True)
            
        if self.args.pooling == 'cls':
            # If using "cls", we add an extra MLP layer
            # (same as BERT's original implementation) over the representation.
            out = out.last_hidden_state[:, 0]  # [batch, 768]
            if is_train:
                return self.mlp(out)
            else:
                return out

        if self.args.pooling == 'cls_before_pooler':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.args.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.args.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

        if self.args.pooling == 'last2_avg':
            second_last = out.hidden_states[-2].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            second_last_avg = torch.avg_pool1d(second_last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((second_last_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

class ScratchModelUnsup(nn.Module):
    """Simcse unsupercised"""

    def __init__(self, *args, **kwargs):
        super(ScratchModelUnsup, self).__init__()
        self.args = kwargs['args']

        pretrained_model = self.args.pretrained_model
        if pretrained_model is None:
            config = BertConfig()
            self.model = BertModel(config, args=self.args)
        elif self.args.arch == 'roberta':
            # pass the config to model constructor instead of from_pretrained
            # this creates the model as per the params in config
            # but with weights randomly initialized
            config = RobertaConfig(
                attention_probs_dropout_prob=self.args.dropout,
                bos_token_id=0,
                eos_token_id=2,
                hidden_act='gelu',
                hidden_dropout_prob=self.args.dropout,
                hidden_size=768,
                initializer_range=0.02,
                intermediate_size=3072,
                layer_norm_eps=1e-05,
                max_position_embeddings=514,
                model_type='roberta',
                num_attention_heads=12,
                num_hidden_layers=12,
                pad_token_id=1,
                vocab_size=50265,
                type_vocab_size=1,
            )
            self.model = RobertaModel(config)
            if self.args.do_mlm:
                self.lm_head = RobertaLMHead(config)
        elif self.args.arch == 'bert':
            # pass the config to model constructor instead of from_pretrained
            # this creates the model as per the params in config
            # but with weights randomly initialized
            config = BertConfig(
                attention_probs_dropout_prob=self.args.dropout,
                gradient_checkpointing=False,
                hidden_act='gelu',
                hidden_dropout_prob=self.args.dropout,
                hidden_size=768,
                initializer_range=0.02,
                intermediate_size=3072,
                layer_norm_eps=1e-12,
                max_position_embeddings=512,
                model_type='bert',
                num_attention_heads=12,
                num_hidden_layers=12,
                pad_token_id=0,
                position_embedding_type='absolute',
                transformers_version='4.6.0.dev0',
                vocab_size=30522,
                type_vocab_size=2,
            )
            self.model = BertModel(config)
            if self.args.do_mlm:
                self.lm_head = BertLMPredictionHead(config)
        else:
            raise ValueError("Unsupported pretrained model")

        self.mlp = MLPLayer(config)

    def forward(self, input_ids, attention_mask, token_type_ids=None, output_hidden_states=True, is_train=True):
        if self.args.arch == 'roberta':
            out = self.model(input_ids, attention_mask, output_hidden_states=output_hidden_states, return_dict=True)
        else:
            out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=output_hidden_states,
                             return_dict=True)

        if self.args.pooling == 'cls':
            # If using "cls", we add an extra MLP layer
            # (same as BERT's original implementation) over the representation.
            out = out.last_hidden_state[:, 0]  # [batch, 768]
            if is_train:
                return self.mlp(out)
            else:
                return out

        if self.args.pooling == 'cls_before_pooler':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.args.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.args.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

        if self.args.pooling == 'last2_avg':
            second_last = out.hidden_states[-2].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            second_last_avg = torch.avg_pool1d(second_last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((second_last_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


'''
    Pooling 
    last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 
    — Sequence of hidden-states at the output of the last layer of the model.

    pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) 
    — Last layer hidden-state of the first token of the sequence (classification token) after 
    further processing through the layers used for the auxiliary pretraining task. E.g. for 
    BERT-family of models, this returns the classification token after processing through a 
    linear layer and a tanh activation function. The linear layer weights are trained from the 
    next sentence prediction (classification) objective during pretraining.

    hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True 
    is passed or when config.output_hidden_states=True) 
    — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an 
    embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

    attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is 
    passed or when config.output_attentions=True) 
    — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, 
    sequence_length, sequence_length).

    Attentions weights after the attention softmax, used to compute the weighted average in 
    the self-attention heads.
'''