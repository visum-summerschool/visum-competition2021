import torch.nn.functional as F
from torch import nn
from transformers import DistilBertModel, DistilBertConfig

from .projection_block import ProjectionBlock


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        pretrained=False,
        finetune=False,
        finetune_start_block=5,
        embedding_size=768,
        projection_size=256,
        dropout=0.1,
        l2_norm=True,
    ):
        """
        Encode text to a fixed size vector

        :param model_name (str): text model name
        :param pretrained (bool): Use pretrained text model (true or false)
        :param finetune (bool): finetune the basemodel layers? (true or false)
        :param finetune_start_block (int): the starting block for finetuning the basemodel
        :param embedding_size (int): text embedding size (output of DistilBertModel)
        :param projection_size (int): projection size (output of ProjectionBlock)
        :param dropout (float): dropout probability used on the ProjectionBlock
        :param l2_norm (bool): apply l2 normalization to the output text embeddings (true or false)
        """
        super().__init__()

        self.l2_norm = l2_norm

        # initialize base model, i.e. DistilBertModel
        if pretrained:
            self.base_model = DistilBertModel.from_pretrained(model_name)
        else:
            self.base_model = DistilBertModel(config=DistilBertConfig())

        # projection block (additional trainable layers)
        self.projection = ProjectionBlock(embedding_size, projection_size, dropout)

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        # fine-tune the basemodel layers?
        self.finetuning(finetune, finetune_start_block)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids (tensor): token ids
        :param attention_mask (tensor):  attention mask
        :return: Text Features
        """
        # base model forward pass
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        last_hidden_state = last_hidden_state[:, self.target_token_idx, :]
        # projection block forward pass
        last_hidden_state = self.projection(last_hidden_state)
        # l2 normalization
        if self.l2_norm:
            last_hidden_state = F.normalize(last_hidden_state, p=2, dim=1)
        return last_hidden_state

    def finetuning(self, finetune=False, finetune_start_block=5):
        """
        Allow or prevent the computation of gradients for the basemodel layers

        :param finetune (bool): perform finetuning?
        :param: finetune_start_block (int): the starting block for finetuning the basemodel
        """
        for p in self.base_model.parameters():
            p.requires_grad = False

        # If fine-tuning, only fine-tune basemodel blocks from finetune_start_block to the last one
        for c in list(self.base_model.children())[-1].layer[finetune_start_block:]:
            for p in c.parameters():
                p.requires_grad = finetune
