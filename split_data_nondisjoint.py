import ast
import os
import math
import random
import numpy as np
import pandas
import argparse

from dataset import train_valid_split


def splitProducts(valuesPerGroup, total, ratio, seed):
    split = math.floor(total * ratio)  # percentage in train
    trainGroups = dict()
    keys = list(valuesPerGroup.keys())
    random.Random(seed).shuffle(keys)

    for g in keys:
        nProducts = sum(trainGroups.values())

        if nProducts == split:
            break
        if nProducts + valuesPerGroup[g] <= split:
            trainGroups[g] = valuesPerGroup[g]

    return set(trainGroups.keys())


def saveToFile(filename, samples):
    f = open(filename, "w")

    for e in samples:
        f.write(str(e) + "\n")

    f.close()
    print("[INFO] Saved; all done!")


def from_np_array(array_string):
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def filter_products(df_products, products):
    return df_products[np.in1d(df_products["productid"], products)]


def filter_outfits(df_outfits, products):
    # transform to dict so that the search is faster
    hashmap = {k: 0 for k in products}
    idx = []
    for outfit in df_outfits["outfit_products"]:
        save = True
        if len(outfit) == 0:
            idx.append(False)
            continue
        for prod in outfit:
            if prod not in hashmap.keys():
                save = False
                break
        idx.append(save)
    return df_outfits[idx]


# copy lines from the df_products
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VISUM split_data_nondisjoint script")
    parser.add_argument(
        "-d",
        "--data_dir",
        default="/home/master/dataset/train",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="processed_data",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "-p",
        "--proportion",
        default=0.7,
        type=float,
        help="train split proportion",
    )
    parser.add_argument(
        "-r",
        "--dry_run",
        default=False,
        action="store_true",
        help="perform a dry run without saving the resulting dataframes",
    )
    args = parser.parse_args()

    # to create the train-val split
    train_folder_name = "train"
    test_folder_name = "valid"

    # load raw df_outfits.csv
    df_outfits = pandas.read_csv(
        os.path.join(args.data_dir, "df_outfits.csv"),
        converters={"outfit_products": from_np_array},
        index_col=0,
    )

    df_products_fn = os.path.join(args.data_dir, "df_products.csv")
    df_products = pandas.read_csv(df_products_fn, index_col=0)

    # train and test split
    train_outfits, test_outfits = train_valid_split(
        df_outfits, val_split=1 - args.proportion, shuffle=True
    )
    train_products = filter_products(
        df_products, list(np.concatenate(train_outfits["outfit_products"]))
    )
    test_products = filter_products(
        df_products, list(np.concatenate(test_outfits["outfit_products"]))
    )

    # print intersection
    kept_outfits_ratio = (len(train_outfits) + len(test_outfits)) / (len(df_outfits))
    kept_products_ratio = (len(train_products) + len(test_products)) / (len(df_outfits))
    intersection_prods = set(train_products["productid"]).intersection(
        set(test_products["productid"])
    )
    intersection_outfit = set(train_outfits["outfit_id"]).intersection(
        set(test_outfits["outfit_id"])
    )
    print("Kept products and outfits:")
    print("\tkept_outfits_ratio", kept_outfits_ratio)
    print("\tkept_products_ratio", kept_products_ratio)
    print("Intersection between train and test:")
    print("\tProducts", len(intersection_prods))
    print("\tOutfits", len(intersection_outfit))

    if not args.dry_run:
        # save cleaned dataframe (df_products)
        print("Saving dataframes...")
        if os.path.isdir(os.path.join(args.out_dir, test_folder_name)) or os.path.isdir(
            os.path.join(args.out_dir, train_folder_name)
        ):
            raise (
                FileNotFoundError(
                    "Target directory already exist. Delete it or change the output directory path."
                )
            )

        os.makedirs(os.path.join(args.out_dir, train_folder_name), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, test_folder_name), exist_ok=True)
        os.makedirs(
            os.path.join(args.out_dir, train_folder_name, "product_images"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(args.out_dir, test_folder_name, "product_images"),
            exist_ok=True,
        )

        train_outfits = train_outfits.reset_index(drop=True)
        test_outfits = test_outfits.reset_index(drop=True)

        train_outfits.to_csv(
            os.path.join(args.out_dir, train_folder_name, "df_outfits.csv")
        )
        test_outfits.to_csv(
            os.path.join(args.out_dir, test_folder_name, "df_outfits.csv")
        )
        train_products.to_csv(
            os.path.join(args.out_dir, train_folder_name, "df_products.csv")
        )
        test_products.to_csv(
            os.path.join(args.out_dir, test_folder_name, "df_products.csv")
        )

        print("Saving train images...")
        for prod in train_products["productid"]:
            src = os.path.join(args.data_dir, "product_images", f"{prod}.jpg")
            dst = os.path.join(
                args.out_dir, train_folder_name, "product_images", f"{prod}.jpg"
            )
            os.symlink(src, dst)

        print("Saving test images...")
        for prod in test_products["productid"]:
            src = os.path.join(args.data_dir, "product_images", f"{prod}.jpg")
            dst = os.path.join(
                args.out_dir, test_folder_name, "product_images", f"{prod}.jpg"
            )
            os.symlink(src, dst)
