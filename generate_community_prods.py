import os
import random

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import itertools

from dataset import load_dataframes
from utils.parsers import ConfigParser
from community import community_louvain


def get_partition(productsInOutfit, seed):
    graph = nx.Graph()
    for prods in productsInOutfit:
        possiblePairs = itertools.combinations(prods, 2)

        for u, v in possiblePairs:
            graph.add_node(u)
            graph.add_node(v)
            if not graph.has_edge(u, v):
                graph.add_edge(u, v)
                graph[u][v]["weight"] = 0
            graph[u][v]["weight"] += 1

    partition = community_louvain.best_partition(graph, random_state=seed)
    return graph, partition


def display_graph(graph, color_map_):
    """
    Display graph

    Args:
        graph (Graph): products graph
        color_map_ (dict): map of colors for the different nodes
    """
    # draw graph
    plt.figure(figsize=(10, 10))
    nx.draw(graph, with_labels=True, node_color=color_map_, node_size=1500, font_size=7)
    plt.show()


def get_color(node, outfit, outfit_id, color):
    """
    Get color for outfit

    Args:
        node (int): product id
        outfit (list): list of products in and outfit
        outfit_id (int): outfitid

    Returns:
        color_map (list): list of colors
    """

    if node in outfit:
        if node not in color_map:
            color_map[node] = str(color[outfit_id])

    return list(color_map.values())


def color_graph(graph, productsInOutfit):
    """Color the nodes from the same outfit

    Args:
        graph (nx.Graph): product graph
        productsInOutfit (list): list of the outfits
    """

    number_of_colors = len(
        df_outfits["outfit_products"]
    )  # number of outfits we are considering
    color = [
        "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
        for i in range(number_of_colors)
    ]

    global color_map
    color_map = {}

    # give color to the nodes in the same outfit
    for i, prods in tqdm(enumerate(productsInOutfit)):
        for node in graph:
            color_map_ = get_color(node, prods, i, color)

    display_graph(graph, color_map_)


def generate_products_graph(config, df_outfits, display=False):
    """
    Gemerate products graph to understand the relations of the products across outfits

    Args:
        df_outfits (Dataframe): dataframe of outfits
        display (bool): display the graph plot (true or false)

    Returns:
        graph (Graph): products graph
    """

    productsInOutfit = df_outfits["outfit_products"].tolist()

    graph, partition = get_partition(productsInOutfit, config.seed)

    if display:
        color_graph(graph, productsInOutfit)

    return graph, partition


def get_partition_dataframe(partition):
    """
    Gemerate partition dataframe [product id/community]

    Args:
        partition (dict): dict (k:v) where k:productid, v: community

    Returns:
        df_partitions (Dataframe): datafrane that represents each product and the community it belongs

    """

    data_tuples = list(zip(partition.keys(), partition.values()))

    df_partitions = pd.DataFrame(data_tuples, columns=["productid", "community"])

    return df_partitions


if __name__ == "__main__":
    # parse config
    cfg = ConfigParser.from_yaml(config_fn="train_config.yaml")

    splits = ["train", "valid"]

    for s in splits:
        print("Generating communities for {} split ...".format(s))

        # load dataframe
        df_outfits, _ = load_dataframes(cfg.DATA, mode=s)
        print("dataset len: {};".format(len(df_outfits)))

        # generate graph
        graph, partition = generate_products_graph(cfg, df_outfits)

        # get partition dataframe
        df_partitions = get_partition_dataframe(partition)

        # save dataframe in cache
        df_community_fn = "df_community_{}.csv".format(s)
        df_partitions.to_csv(os.path.join(cfg.DATA.cache_dir, df_community_fn))
        print("# of communities: ", len(df_partitions.community.unique()))
