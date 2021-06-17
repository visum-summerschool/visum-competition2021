"""
Script to generate the test set queries
 * For each outfit generate sets of candidates with 1 positive and 4 negatives
"""
import pandas
import os
import random
import numpy as np
from os.path import join
import argparse

from split_data_nondisjoint import from_np_array

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VISUM generate_test_queries script")
    parser.add_argument(
        "-d",
        "--data_dir",
        default="processed_data/valid/",
        type=str,
        help="data directory",
    )
    args = parser.parse_args()

    df_outfits_fn = os.path.join(args.data_dir, "df_outfits.csv")
    df_outfits = pandas.read_csv(
        df_outfits_fn, converters={"outfit_products": from_np_array}, index_col=0
    )
    df_products = pandas.read_csv(
        os.path.join(args.data_dir, "df_products.csv"), index_col=0
    )

    query_outfits = list()
    query_options = list()
    query_solutions = list()

    for outfit in df_outfits.iterrows():
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


pandas.DataFrame.from_dict({"outfit_products": query_outfits}).to_csv(
    join(args.data_dir, "queries.csv")
)
pandas.DataFrame.from_dict({"productid": query_solutions}).to_csv(
    join(args.data_dir, "solutions.csv")
)
pandas.DataFrame.from_dict({"productids": query_options}).to_csv(
    join(args.data_dir, "options.csv")
)
