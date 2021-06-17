from torch import nn

from .modules.image_encoder import ImageEncoder
from .modules.multimodal_encoder import SimpleMultimodalEncoder
from .modules.text_encoder import TextEncoder


class Model(nn.Module):
    def __init__(self, input_modality, cfg_image, cfg_text, cfg_mm):
        """
        :param input_modality (str): input modality (image, text or multimodal)
        :param cfg_image (ConfigParser object): Image encoder options
        :param cfg_text (ConfigParser object): Text encoder options
        :param cfg_mm (ConfigParser object): Multimodal encoder options
        """
        super().__init__()
        self.input_modality = input_modality

        if self.input_modality == "image":
            self.image_encoder = ImageEncoder(
                cfg_image.model_name,
                cfg_image.pretrained,
                cfg_image.finetune,
                cfg_image.finetune_start_block,
                cfg_image.embedding_size,
                cfg_image.projection_size,
                cfg_image.dropout,
                cfg_image.l2_norm,
            )

        elif self.input_modality == "text":
            self.text_encoder = TextEncoder(
                cfg_text.model_name,
                cfg_text.pretrained,
                cfg_text.finetune,
                cfg_text.finetune_start_block,
                cfg_text.embedding_size,
                cfg_text.projection_size,
                cfg_text.dropout,
                cfg_text.l2_norm,
            )

        elif self.input_modality == "multimodal":
            self.image_encoder = ImageEncoder(
                cfg_image.model_name,
                cfg_image.pretrained,
                cfg_image.finetune,
                cfg_image.finetune_start_block,
                cfg_image.embedding_size,
                cfg_image.projection_size,
                cfg_image.dropout,
                cfg_image.l2_norm,
            )

            self.text_encoder = TextEncoder(
                cfg_text.model_name,
                cfg_text.pretrained,
                cfg_text.finetune,
                cfg_text.finetune_start_block,
                cfg_text.embedding_size,
                cfg_text.projection_size,
                cfg_text.dropout,
                cfg_text.l2_norm,
            )

            self.multimodal_encoder = SimpleMultimodalEncoder(
                cfg_text.projection_size,
                cfg_image.projection_size,
                cfg_mm.embedding_size,
                cfg_mm.dropout,
                cfg_mm.l2_norm,
            )
        else:
            raise AssertionError("Invalid INPUT_MODALITY!")

    def forward(self, x):
        """
        :param x (dict): multimodal input dictionary (image, input_ids, attention_mask)
        :return image_features (tensor), text_features(tensor), multimodal_features (tensor): image, text and
                            multimodal features
        """
        if self.input_modality == "image":
            return self.image_encoder.forward(x["image"])

        elif self.input_modality == "text":
            return self.text_encoder.forward(
                input_ids=x["input_ids"], attention_mask=x["attention_mask"]
            )

        elif self.input_modality == "multimodal":
            image_features = self.image_encoder.forward(x["image"])
            text_features = self.text_encoder.forward(
                input_ids=x["input_ids"], attention_mask=x["attention_mask"]
            )
            return self.multimodal_encoder.forward(text_features, image_features)

        else:
            raise AssertionError("Invalid INPUT_MODALITY!")
