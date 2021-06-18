import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer

from dataset import load_dataframes, FITBDataset
from utils.parsers import ConfigParser
from utils.utils import from_np_array, fix_random_seeds
from utils.visualizer import FITB_random_query_viz, FITB_query_viz
from models.baseline import Model


def get_scores(query_embeddings, candidates_embeddings, aggregation_foo="sum"):
    """Get candidates scores

    Args:
       query_embeddings (torch.tensor): query products embeddings. Shape: (num_query_prods, emb_size).
       candidates_embeddings (torch.tensor): candidates products embeddings. Shape: (num_candidate_prods, emb_size).
       aggregation_foo (str): type of the aggregation function (sum or min)

    Returns:
       candidates_scores (numpy array): array of scores for each candidate

    """

    numb_candidates = candidates_embeddings.size(0)
    numb_query = query_embeddings.size(0)
    candidates_scores = np.zeros(numb_candidates, dtype=np.float32)
    for candidate_idx in range(numb_candidates):
        candidate_emb = candidates_embeddings[candidate_idx].unsqueeze(0)

        if aggregation_foo == "sum":
            score = 0.0
        elif aggregation_foo == "min":
            score = float("inf")
        else:
            raise ValueError("Invalid aggregation function!")

        for query_idx in range(numb_query):
            query_emb = query_embeddings[query_idx].unsqueeze(0)

            if aggregation_foo == "sum":
                score += F.pairwise_distance(candidate_emb, query_emb, 2)
            elif aggregation_foo == "min":
                score = min(score, F.pairwise_distance(candidate_emb, query_emb, 2))
            else:
                raise ValueError("Invalid aggregation function!")

        candidates_scores[candidate_idx] = score.squeeze().cpu().numpy()

    return candidates_scores


def main(config, test_dir, out_dir, agg_foo="sum", mode="valid"):
    # load test split dataframes
    _, df_products_test = load_dataframes(config.DATA, mode=mode, load_outfits=False)

    # load FITB test queries: queries.csv and options.csv dataframes
    df_candidates = pd.read_csv(
        os.path.join(test_dir, "options.csv"),
        converters={"productids": from_np_array},
        index_col=0,
    )

    df_queries = pd.read_csv(
        os.path.join(test_dir, "queries.csv"),
        converters={"outfit_products": from_np_array},
        index_col=0,
    )

    # initialize text tokenizer
    text_tokenizer = DistilBertTokenizer.from_pretrained(config.TOKENIZER.path)

    # FITB dataset
    fitb_dataset = FITBDataset(
        config, df_queries, df_candidates, df_products_test, text_tokenizer, mode=mode
    )

    # load model checkpoint
    model = Model(
        config.INPUT_MODALITY,
        config.IMAGE_ENCODER,
        config.TEXT_ENCODER,
        config.MULTIMODAL_ENCODER,
    )

    model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(model)

    # iterate over test FITB queries
    preds = []
    preds_ranked = []
    for sample in tqdm(fitb_dataset, desc="Performing FITB inference"):
        # unpack data
        query = {k: v.to(device) for k, v in sample["query"].items()}
        candidates = {k: v.to(device) for k, v in sample["candidates"].items()}
        candidates_ids = candidates["productids"].cpu().numpy()

        # forward pass
        with torch.no_grad():
            query_emb = model(query)
            candidates_emb = model(candidates)

        # get a FITB score for each candidate product
        candidates_scores = get_scores(
            query_emb, candidates_emb, aggregation_foo=agg_foo
        )

        # get the predicted candidate product id (i.e. the one with the lowest score - distance)
        pred_id = candidates_ids[np.argmin(candidates_scores)]
        preds.append(pred_id)
        preds_ranked.append(candidates_ids[np.argsort(candidates_scores)])

    # save predictions
    pd.DataFrame.from_dict({"productid": preds}).to_csv(
        os.path.join(out_dir, "preds.csv")
    )
    pd.DataFrame.from_dict({"productid": preds_ranked}).to_csv(
        os.path.join(out_dir, "preds_ranked.csv")
    )


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="VISUM test script")
    parser.add_argument(
        "-c",
        "--config",
        default="test_config.yaml",
        type=str,
        help="config file name",
    )
    parser.add_argument(
        "-f",
        "--agg_foo",
        default="sum",
        type=str,
        help="aggregation function of the distances between candidates and query products(sum or min)",
    )
    parser.add_argument(
        "-d", "--display", help="display FITB results", action="store_true"
    )
    parser.add_argument(
        "-t",
        "--test_dir",
        default="/home/master/dataset/test",
        type=str,
        help="location of test data",
    )
    parser.add_argument(
        "-o", "--out_dir", default="", type=str, help="output directory"
    )

    args = parser.parse_args()

    # parse config file
    config = ConfigParser.from_yaml(config_fn=args.config, mode="test")

    # fixing random seed for reproducibility
    fix_random_seeds(config.seed)

    # FITB queries inference
    main(config, args.test_dir, args.out_dir, args.agg_foo)

    # Visualize FITB results
    if args.display:
        df_queries = pd.read_csv(
            os.path.join(args.test_dir, "queries.csv"),
            converters={"outfit_products": from_np_array},
            index_col=0,
        )
        df_preds = pd.read_csv(
            os.path.join(args.out_dir, "preds_ranked.csv"),
            converters={"productid": from_np_array},
            index_col=0,
        )
        df_gt = pd.read_csv(
            os.path.join(args.test_dir, "solutions.csv"),
            converters={"productid": from_np_array},
            index_col=0,
        )

        FITB_random_query_viz(
            config, df_queries, df_preds, df_gt, save_path=args.out_dir
        )
        # FITB_query_viz(config, df_queries, df_preds, df_gt, save_path=args.out_dir)
