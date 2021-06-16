"""
Script to generate the test set queries
 * For each outfit generate sets of candidates with 1 positive and 4 negatives
"""
import pandas
import os
import random
import numpy as np
from os.path import join

from split_data_nondisjoint import from_np_array
from utils.parsers import ConfigParser

if __name__ == "__main__":

    cfg = ConfigParser.from_yaml(config_fn="train_config.yaml")
    dataDir = "processed_data/valid/"

    df_outfits_fn = os.path.join(dataDir, "df_outfits.csv")
    df_outfits = pandas.read_csv(
        df_outfits_fn, converters={"outfit_products": from_np_array}, index_col=0
    )
    df_products = pandas.read_csv(os.path.join(dataDir, "df_products.csv"), index_col=0)

    query_outfits = list()
    query_options = list()
    query_solutions = list()

    numb_test_queries = (
        cfg.DATA.num_valid_outfits
        if cfg.DATA.num_valid_outfits is not None
        else len(df_outfits)
    )
    for o_idx, outfit in enumerate(df_outfits.iterrows()):
        if len(outfit[1]["outfit_products"]) == 0:
            continue

        prods = outfit[1]["outfit_products"]
        idx = random.randint(
            0, len(prods) - 1
        )  # closed interval on random.randint func
        prod = prods[idx]
        prods = np.delete(prods, idx)

        prod_data = df_products[df_products["productid"] == prod]
        cat = prod_data["category"].item()

        c = 0
        try:
            candidates = df_products[df_products["category"] == cat][
                df_products["productid"] != prod
            ].sample(3)
            candidates = np.concatenate((np.array(candidates["productid"]), [prod]))
            np.random.shuffle(candidates)
            query_outfits.append(prods)
            query_solutions.append(prod)
            query_options.append(candidates)
        except ValueError:
            candidates = None

        if o_idx > numb_test_queries:
            break

pandas.DataFrame.from_dict({"outfit_products": query_outfits}).to_csv(
    join(dataDir, "queries.csv")
)
pandas.DataFrame.from_dict({"productid": query_solutions}).to_csv(
    join(dataDir, "solutions.csv")
)
pandas.DataFrame.from_dict({"productids": query_options}).to_csv(
    join(dataDir, "options.csv")
)
