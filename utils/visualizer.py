import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg

from utils import tokens_str


def triplet_outfit_visualizer(
    sample, inv_transform, word_map, category_map=None, save_fn=None
):
    """Display a sample of the TripletOutfitDataset"""

    numb_prods = len(sample)
    inv_category_map = {v: k for k, v in category_map.items()}
    for i, (k, v) in enumerate(sample.items()):
        img = np.transpose(inv_transform(v["image"]), (1, 2, 0))
        desc = tokens_str(v["input_ids"], word_map)
        cat = inv_category_map[int(v["category"])]

        plt.subplot(1, numb_prods, i + 1)
        plt.imshow(img)
        plt.grid(False)
        plt.axis("off")
        plt.title("{} \n category: {}".format(k, cat))
        print(desc)
        print("-")
    if save_fn is None:
        plt.show()
    else:
        plt.savefig(save_fn)


def display_outfit_from_batch(batch, sample_index, word_map, category_map):
    """Display outfit info from batch"""

    # get outfit sample from batch
    sample = {k: v[sample_index] for k, v in batch.items()}
    pad_mask = sample["pad_mask"] == 1

    # display prod info
    inv_category_map = {v: k for k, v in category_map.items()}
    outfit_len = len(sample["imgs"][pad_mask])
    for p in range(outfit_len):
        plt.subplot(1, outfit_len, p + 1)
        plt.imshow(np.transpose(sample["imgs"][p], (1, 2, 0)))
        plt.title(inv_category_map[int(sample["cats"][p])])
        plt.axis("off")
        print("-")
        # print(sample['text_tokens'][p])
        print(tokens_str(sample["text_tokens"][p], word_map))
        print("-")
    plt.show()


def display_training_curves(training_data, trainning_labels, cfg):
    """
    Saves image of the training curves

    Args:
        training_data (list): list of dict with training values
        training labels (list): list of labels for each dict in the training data
    """

    linestyles = ["-", "--"]
    color = ["#FFDDE2FF", "#FAA094FF", "#9ED9CCFF", "#008C76FF"]
    label_modality = ["image", "text", "multimodal (UMD)", "multimodal (simple)"]
    train_curves_fn = os.path.join(
        cfg.TRAINER.save_dir,
        "training_curves/modality_{}.png".format(len(training_data)),
    )

    for idx, dict in enumerate(training_data):
        for i, l in enumerate(trainning_labels[idx]):
            plt.plot(
                training_data[idx][l],
                label=str(l) + " " + label_modality[idx],
                linestyle=linestyles[i],
                color=color[idx],
            )

    # draw
    plt.rcParams["figure.figsize"] = [20, 20]
    plt.title("Learning curves")
    plt.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.rc("grid", linestyle="-", color="#EFF0F1")
    plt.grid(True)
    plt.savefig(train_curves_fn)


def draw_rectangle(img, start_point, end_point, color):
    """Draw rectangle around the image to differentiate the correct selection from our prediction

    Args:
        img (ndarray): image
        start_point (tuple): coordinates of the starting point
        end_point (tuple): coordinates of the ending point
        color (tuple): rgb color for the rectangle

    Returns:
        rectangle for the given image
    """

    image = cv2.rectangle(img, start_point, end_point, color, 3)
    return image


def display_fitb_query(query, predictions, ground_truth, images_dir, save_fn=None):
    """Display fitb query and model candidates predictions

    Args:
        query (ndarray): outfit query
        predictions (ndarray): ranked candidates
        ground_truth (ndarray): correct product selection
        images_dir (str): path to image directory
        save_fn (str): path to directory to save generated figures
    """

    # display query products (outfit)
    num_query = len(query)
    for i, p in enumerate(query):
        # read product image
        im_fn = os.path.join(images_dir, str(p) + ".jpg")
        im = mpimg.imread(im_fn)

        # imshow
        plt.subplot(1, num_query, i + 1)
        plt.imshow(im)
        plt.title(str(p))
        plt.axis("off")
    plt.suptitle("Outfit query")
    if save_fn is None:
        plt.show()
    else:
        plt.savefig(save_fn.format("query"))

    # display model candidates predictions
    num_preds = len(predictions)
    for i, prod_id in enumerate(predictions):
        # read product image
        im_fn = os.path.join(images_dir, str(prod_id) + ".jpg")
        im = mpimg.imread(im_fn)
        start_point = (0, im.shape[0])
        end_point = (im.shape[1], 0)
        if i == 0 or query[0] == ground_truth:
            im = draw_rectangle(im, start_point, end_point, (0, 0, 255))

        elif prod_id == ground_truth:
            im = draw_rectangle(im, start_point, end_point, (0, 255, 0))

        # imshow
        plt.subplot(1, num_preds, i + 1)
        plt.imshow(im)
        plt.title(str(prod_id))
        plt.axis("off")
    plt.suptitle("Ranked Candidates")
    if save_fn is None:
        plt.show()
    else:
        plt.savefig(save_fn.format("ranked_candidates"))


def FITB_query_viz(config, df_queries, df_predictions, df_solutions):
    """
    Display query and prediction

    Args:
        config: config file
        df_queries (dataframe): dataframe of the query products
        df_predictions (dataframe): dataframe of the ranked predictions
        df_solutions (dataframe): dataframe of the solutions (ground truth)
    """

    images_dir = config.DATA.product_imgs_dir
    fitb_dir = os.path.join(config.TRAINER.save_dir, "FITB_images")
    os.makedirs(fitb_dir, exist_ok=True)

    # display candidates with green square on the correct and blue on our prediction
    for idx, (query, predictions, ground_truth) in enumerate(
        zip(
            df_queries["outfit_products"],
            df_predictions["productid"],
            df_solutions["productid"],
        )
    ):
        save_fn = os.path.join(fitb_dir, str(idx) + "_{}.png")
        display_fitb_query(query, predictions, ground_truth, images_dir, save_fn)


def FITB_random_query_viz(
    config, df_queries, df_predictions, df_solutions, n=3, column="productid"
):
    """
     Display random n correct query and prediction and random n incorrect query and prediction

    Args:
        config: config file
        df_queries (Dataframe): dataframe of the outfits query
        df_predictions (Dataframe): dataframe with the model candidates predictions
        df_solutions (Dataframe): dataframe with the solutions
        n (int): number of samples we want to generate
        column (str): name of the column in the dataframe

    """

    fitb_mask = np.array(
        [p[0] == s for p, s in zip(df_predictions[column], df_solutions[column])]
    )
    images_dir = config.DATA.product_imgs_dir

    # correct preds
    df_predictions_correct = df_predictions[fitb_mask].sample(n)
    fitb_mask_idx = df_predictions_correct.index
    df_solutions_correct = df_solutions.iloc[fitb_mask_idx]
    df_queries_correct = df_queries.iloc[fitb_mask_idx]

    # display candidates with green square on the correct and blue on our prediction
    for idx, (query, predictions, ground_truth) in enumerate(
        zip(
            df_queries_correct["outfit_products"],
            df_predictions_correct[column],
            df_solutions_correct[column],
        )
    ):
        display_fitb_query(query, predictions, ground_truth, images_dir)

    # incorrect preds
    df_predictions_incorrect = df_predictions[~fitb_mask].sample(n)
    fitb_mask_inc_idx = df_predictions_incorrect.index
    df_solutions_incorrect = df_solutions.iloc[fitb_mask_inc_idx]
    df_queries_incorrect = df_queries.iloc[fitb_mask_inc_idx]

    # display candidates with green square on the correct and blue on our prediction
    for idx, (query, predictions, ground_truth) in enumerate(
        zip(
            df_queries_incorrect["outfit_products"],
            df_predictions_incorrect[column],
            df_solutions_incorrect[column],
        )
    ):
        display_fitb_query(query, predictions, ground_truth, images_dir)
