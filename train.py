import argparse

import torch
from torch.nn import TripletMarginLoss
from transformers import DistilBertTokenizer

from dataset import load_dataframes, build_loaders
from models.baseline import Model
from trainer import Trainer
from utils.parsers import ConfigParser
from utils.utils import fix_random_seeds


def main(config):
    # load dataframe and get train and valid splits
    df_outfits_train, df_products_train = load_dataframes(config.DATA, mode="train")
    df_outfits_valid, df_products_valid = load_dataframes(config.DATA, mode="valid")
    print(
        "train_dataset len: {}; valid_dataset len: {} ".format(
            len(df_outfits_train), len(df_outfits_valid)
        )
    )

    # initialize text tokenizer
    text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    text_tokenizer.save_pretrained(
        config.TOKENIZER.save_path.format(config.TRAINER.time_stamp)
    )

    # build dataloaders
    train_loader = build_loaders(
        config,
        df_outfits_train,
        df_products_train,
        text_tokenizer,
        mode="train",
    )
    valid_loader = build_loaders(
        config,
        df_outfits_valid,
        df_products_valid,
        text_tokenizer,
        mode="valid",
    )
    print(
        "train_loader len: {}; valid_loader len: {} ".format(
            len(train_loader), len(valid_loader)
        )
    )

    # initialize model
    model = Model(
        config.INPUT_MODALITY,
        config.IMAGE_ENCODER,
        config.TEXT_ENCODER,
        config.MULTIMODAL_ENCODER,
    )
    numb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print("Trainable parameters: {}".format(numb_params))

    # prepare for GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # select loss function and optimizer
    loss_fn = TripletMarginLoss(margin=config.LOSS.margin, p=2)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.OPTIMIZER.lr,
        weight_decay=config.OPTIMIZER.weight_decay,
    )

    # instantiate trainer and train the model
    trainer = Trainer(
        config,
        model,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        device,
        metrics=[],
        scheduler=None,
    )

    trainer.fit()
    train_history = trainer.get_history()
    print(train_history)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="VISUM train script")
    parser.add_argument(
        "-c",
        "--config",
        default="train_config.yaml",
        type=str,
        help="config file name",
    )

    args = parser.parse_args()

    # parse config file
    config = ConfigParser.from_yaml(config_fn=args.config, mode="train")

    # fixing random seed for reproducibility
    fix_random_seeds(config.seed)

    # train
    main(config)
