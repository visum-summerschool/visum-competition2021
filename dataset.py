import math
import os
import pickle
from abc import abstractmethod
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transforms import Transforms
from utils.utils import from_np_array, check_isrgb, arraystr_to_array, cat2id


class BaseDataset(Dataset):
    """Base class for all datasets (TripletOutfitDataset and FITBDataset)"""

    def __init__(self, cfg, df_products, text_tokenizer, mode="train"):

        self.cfg = cfg
        self.df_products = df_products
        self.text_tokenizer = text_tokenizer
        self.mode = mode

        # get helper category mapping dicts
        self.category2id = cat2id(self.df_products)

        # get transforms
        self.transforms = Transforms(cfg=self.cfg.TRANSFORMS).get_transforms(
            mode=self.mode
        )

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError("getitem method is not implemented!")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("len method is not implemented!")

    def get_product_data(self, productid, columns=["category", "description"]):
        """Get product data for a given productid"""

        # get product data
        prod_info = self.df_products[self.df_products["productid"] == productid]

        # read product image
        image_fn = os.path.join(
            self.cfg.DATA.product_imgs_dir, str(productid) + self.cfg.DATA.imgs_ext
        )
        prod_img = check_isrgb(io.imread(image_fn))

        # organize product data in a dict
        product_data = {c: prod_info[c].iloc[0] for c in columns}
        product_data["image"] = prod_img

        # replace nan product descriptions with the corresponding product name
        if not isinstance(product_data["description"], str) and math.isnan(
            product_data["description"]
        ):
            product_data["description"] = prod_info["productname"].iloc[0]

        return product_data

    def transform(self, product_sample):
        """Apply transformations to a given product data sample"""

        transform_data = {}

        # apply transforms to product image
        transform_data["image"] = self.transforms(product_sample["image"])

        # text tokenization
        text_tokens = self.text_tokenizer(
            product_sample["description"],
            padding=True,
            truncation=True,
            max_length=self.cfg.TOKENIZER.max_length,
        )

        text_tokens_pad = {
            k: self.pad_tokens(v, self.cfg.TOKENIZER.max_length)
            for k, v in text_tokens.items()
        }
        transform_data.update({k: torch.tensor(v) for k, v in text_tokens_pad.items()})

        # get category id
        transform_data["category"] = torch.tensor(
            self.category2id[product_sample["category"]]
        )

        return transform_data

    def pad_tokens(self, token_list, max_length, pad_token=0):
        """pad a list of tokens (token_list) with pad_token"""
        return token_list + [pad_token] * (max_length - len(token_list))


class TripletOutfitDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        df_outfits,
        df_products,
        text_tokenizer,
        mode="train",
        anchor_pos_pairs_cache_fn="anchor_pos_pairs_{}.pickle",
        category_2_prod_cache_fn="category_2_prod_{}.pickle",
        df_community_fn="df_community_{}.csv",
        neg_attempts=200,
    ):
        super().__init__(cfg, df_products, text_tokenizer, mode)

        self.df_outfits = df_outfits
        self.anchor_pos_pairs_cache_fn = anchor_pos_pairs_cache_fn.format(self.mode)
        self.category_2_prod_cache_fn = category_2_prod_cache_fn.format(self.mode)
        self.df_community_fn = df_community_fn.format(self.mode)
        self.neg_attempts = neg_attempts

        # get helper category mapping dicts
        self.category_2_prod = self.get_category_mapping(
            self.df_outfits, self.df_products
        )

        # pairwise anchor-positive pairs in df_outfits. negatives are sampled by the __getitem__ method
        self.anchor_pos_pairs = self.gen_anchor_positive_pairs()

        # load communities dataframe (needed for negative sampling) and convert it to dict for faster access
        df_communities = pd.read_csv(
            os.path.join(self.cfg.DATA.cache_dir, self.df_community_fn),
            converters={"community_prods": arraystr_to_array},
            index_col=0,
        )
        self.community_dict = dict(
            zip(df_communities.productid, df_communities.community)
        )

    def __getitem__(self, index):
        # get anchor-positive sample
        ap_sample = self.anchor_pos_pairs[index]
        outfit_id, anchor_id, pos_id = (
            ap_sample["outfit_id"],
            ap_sample["anchor_id"],
            ap_sample["positive_id"],
        )

        # get anchor-pos data
        anchor = self.get_product_data(anchor_id)
        pos = self.get_product_data(pos_id)

        # get a negative sample
        neg_id = self.negative_sampling(outfit_id, pos_id, pos["category"], anchor_id)
        neg = self.get_product_data(neg_id)

        # apply data transformations
        if self.transforms is not None:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return {"anchor": anchor, "pos": pos, "neg": neg}

    def __len__(self):
        return len(self.anchor_pos_pairs)

    def negative_sampling(self, outfit_id, pos_id, pos_category, anchor_id):
        """Get a negative sample. A negative sample should belong to same exact category of the positive sample, but
        should be from a different community"""

        # get community of anchor and positive samples
        pos_community = self.community_dict[pos_id]
        anchor_community = self.community_dict[anchor_id]

        # negative sampling
        neg_id = pos_id
        neg_outfit = outfit_id
        neg_community = pos_community
        outfit_candidates = list(self.category_2_prod[pos_category].keys())
        attempts = 0
        while (
            neg_id == pos_id  # negative and positive product_ids have to be different
            or neg_outfit
            == outfit_id  # negative and positive samples have to be from different outfits
            or pos_community
            == neg_community  # negative and positive samples have to be from different communities
            or anchor_community == neg_community
        ):  # negative and anchor samples have to be from different communities

            if attempts > self.neg_attempts:
                break

            neg_outfit = np.random.choice(outfit_candidates)
            neg_candidates = self.category_2_prod[pos_category][neg_outfit]
            neg_id = np.random.choice(neg_candidates)
            neg_community = self.community_dict[neg_id]
            attempts += 1

        return neg_id

    def gen_anchor_positive_pairs(self):
        """generate a mapping dict (anchor_pos_pairs) with all pairwise anchor-positive combinations"""

        # load anchor_pos_pairs if it already exists in cache
        file_fn = os.path.join(self.cfg.DATA.cache_dir, self.anchor_pos_pairs_cache_fn)
        if os.path.isfile(file_fn):
            print("Loading anchor_pos_pairs from: {}".format(file_fn))
            with open(file_fn, "rb") as handle:
                return pickle.load(handle)

        # gen anchor_pos_pairs dict
        anchor_pos_pairs = {}
        sample_index = 0
        for _, row in tqdm(
            self.df_outfits.iterrows(),
            desc="Generating anchor_pos_pairs ...",
            total=len(self.df_outfits),
        ):

            # ignore outfits with single products
            if len(row["outfit_products"]) < 2:
                continue

            ap_combo = list(combinations(row["outfit_products"], 2))
            for anchor, positive in ap_combo:
                anchor_pos_pairs[sample_index] = {
                    "outfit_id": row["outfit_id"],
                    "anchor_id": anchor,
                    "positive_id": positive,
                }
                sample_index += 1

        # save anchor_pos_pairs in cache
        with open(file_fn, "wb") as handle:
            pickle.dump(anchor_pos_pairs, handle)

        return anchor_pos_pairs

    def get_category_mapping(self, df_outfits, df_products):
        """generate a dict (category_2_prod) with a mapping from category-to-outfit_id-to-product_id, in order to
        speed up the negative sampling process."""

        # load anchor_pos_pairs if it already exists in cache
        file_fn = os.path.join(self.cfg.DATA.cache_dir, self.category_2_prod_cache_fn)
        if os.path.isfile(file_fn):
            print("Loading category_2_prod from: {}".format(file_fn))
            with open(file_fn, "rb") as handle:
                return pickle.load(handle)

        # gen category_2_prod dict
        category_2_prod = {}
        for _, row in tqdm(
            df_outfits.iterrows(),
            desc="Generating category_2_prod dict ...",
            total=len(df_outfits),
        ):
            # ignore outfits with single products
            if len(row["outfit_products"]) < 2:
                continue

            outfit_id = row["outfit_id"]
            for product_id in row["outfit_products"]:
                category = df_products[df_products["productid"] == product_id][
                    "category"
                ].iloc[0]

                if category not in category_2_prod:
                    category_2_prod[category] = {}

                if outfit_id not in category_2_prod[category]:
                    category_2_prod[category][outfit_id] = []

                category_2_prod[category][outfit_id].append(product_id)

        # save anchor_pos_pairs in cache
        with open(file_fn, "wb") as handle:
            pickle.dump(category_2_prod, handle)

        return category_2_prod


class FITBDataset(BaseDataset):
    def __init__(
        self, cfg, df_queries, df_candidates, df_products, text_tokenizer, mode="test"
    ):
        super().__init__(cfg, df_products, text_tokenizer, mode)
        self.df_queries = df_queries
        self.df_candidates = df_candidates

    def __getitem__(self, index):

        # get query products
        query_ids = self.df_queries.iloc[index]["outfit_products"]
        query_prods = [self.get_product_data(pid) for pid in query_ids]

        # get options/candidates products
        candidate_ids = self.df_candidates.iloc[index]["productids"]
        candidate_prods = [self.get_product_data(pid) for pid in candidate_ids]

        # apply transforms
        if self.transforms is not None:
            query_prods = [self.transform(p) for p in query_prods]
            candidate_prods = [self.transform(p) for p in candidate_prods]

        # create a FITB item
        query_prods = {
            k: torch.stack([d[k] for d in query_prods]) for k in query_prods[0].keys()
        }
        candidate_prods = {
            k: torch.stack([d[k] for d in candidate_prods])
            for k in candidate_prods[0].keys()
        }

        query_prods.update({"productids": torch.tensor(query_ids)})
        candidate_prods.update({"productids": torch.tensor(candidate_ids)})

        fitb_item = {
            "query": query_prods,
            "candidates": candidate_prods,
            "query_size": len(query_prods["category"]),
        }

        return fitb_item

    def __len__(self):
        return len(self.df_queries)


def load_dataframes(cfg, mode="train", load_outfits=True):
    if load_outfits:
        # load df_outfits dataframe
        df_outfits = pd.read_csv(
            cfg.df_outfits_fn.format(mode),
            converters={"outfit_products": from_np_array},
            index_col=0,
        )
    else:
        df_outfits = None

    # load df_products dataframe
    df_products = pd.read_csv(cfg.df_products_fn.format(mode), index_col=0)

    return df_outfits, df_products


def train_valid_split(df_outfits, val_split=0.2, shuffle=True):
    np.random.seed(42)

    # df_outfit indices
    df_outfits = df_outfits.reset_index(drop=True)
    indices = np.arange(len(df_outfits))

    if shuffle:
        np.random.shuffle(indices)

    # get train and valid indices
    train_size = int(1 - val_split * len(df_outfits))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    # get train and valid dataframes
    df_outfits_train = df_outfits.loc[train_indices].reset_index(drop=True)
    df_outfits_valid = df_outfits.loc[valid_indices].reset_index(drop=True)

    return df_outfits_train, df_outfits_valid


def build_loaders(cfg, df_outfits, df_products, text_tokenizer, mode):
    """Build dataloader"""

    dataset = TripletOutfitDataset(cfg, df_outfits, df_products, text_tokenizer, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATA.batch_size,
        num_workers=cfg.DATA.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader
