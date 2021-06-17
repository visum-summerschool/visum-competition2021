import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .projection_block import ProjectionBlock


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name="resnet50",
        pretrained=False,
        finetune=False,
        finetune_start_block=7,
        embedding_size=2048,
        projection_size=256,
        dropout=0.1,
        l2_norm=True,
    ):
        """
        Encode text to a fixed size vector

        :param model_name (str): Image model to be used
        :param pretrained (bool): Use pretrained image model (true or false)
        :param finetune (bool): finetune the basemodel layers? (true or false)
        :param finetune_start_block (int): the starting block for finetuning the basemodel
        :param embedding_size (int): image embedding size (output of resnet50)
        :param projection_size (int): projection size (output of ProjectionBlock)
        :param dropout (float): dropout probability used on the ProjectionBlock
        :param l2_norm (bool): apply l2 normalization to the output image embeddings (true or false)
        """
        super().__init__()

        self.l2_norm = l2_norm

        # initialize base model for feature extraction (ResNet50), and remove the last linear layer
        self.base_model = models.resnet50(pretrained=pretrained)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        # initialize a projection block (additional trainable layers)
        self.projection = ProjectionBlock(embedding_size, projection_size, dropout)

        # fine-tune the basemodel layers?
        self.finetuning(finetune, finetune_start_block)

    def forward(self, image):
        """
        :param image (torch.tensor): images (batch_size, 3, image_size, image_size)
        :return: output (torch.tensor): image features (batch_size, projection_size)
        """
        # base model forward pass
        output = self.base_model(image)
        output = torch.squeeze(output, 2)
        output = torch.squeeze(output, 2)
        # projection block forward pass
        output = self.projection(output)
        # l2 normalization
        if self.l2_norm:
            output = F.normalize(output, p=2, dim=1)
        return output

    def finetuning(self, finetune=False, finetune_start_block=5):
        """
        Allow or prevent the computation of gradients for the basemodel layers

        :param finetune (bool): perform finetuning?
        :param: finetune_start_block (int): the starting block for finetuning the basemodel
        """
        for p in self.base_model.parameters():
            p.requires_grad = False

        # If fine-tuning, only fine-tune basemodel blocks from finetune_start_block to the last one
        for c in list(self.base_model.children())[finetune_start_block:]:
            for p in c.parameters():
                p.requires_grad = finetune
