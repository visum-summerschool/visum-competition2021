import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.modules.projection_block import ProjectionBlock


class SimpleMultimodalEncoder(nn.Module):

    def __init__(self, text_embedding_sz, image_embedding_sz, joint_embedding_sz, dropout=0.1, l2_norm=True):
        """
        :param text_embedding_sz (int): Text embedding size
        :param image_embedding_sz (int): Image embedding size
        :param joint_embedding_sz (int): Joint embeddings size (Text + Image)
        :param dropout (float): dropout rate
        """
        super().__init__()

        self.l2_norm = l2_norm

        # initialize a projection block to fuse image and text embeddings
        self.projection = ProjectionBlock(text_embedding_sz + image_embedding_sz, joint_embedding_sz, dropout)

    def forward(self, text_embedding, img_embedding):
        """
        :param text_embedding (tensor): Text embeddings
        :param img_embedding (tensor): Image embeddings
        :return output (tensor): Multimodal embeddings
        """

        # concat text and image modalities
        output = torch.cat((text_embedding, img_embedding), 1)

        # fuse both modalities thought a projection block
        output = self.projection(output)

        # l2 normalization
        if self.l2_norm:
            output = F.normalize(output, p=2, dim=1)
        return output
