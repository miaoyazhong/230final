from typing import Tuple
import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer
import os
import tensorflow as tf
# -------------------------------------------------------

# Configuration

# -------------------------------------------------------

IMG_SIZE = (200, 320)
MAX_LEN = 64
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# -------------------------------------------------------

# Image preprocessing

# -------------------------------------------------------

def load_and_process_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize to [0,1]
    return img
# -------------------------------------------------------

# Text preprocessing

# -------------------------------------------------------

def tokenize_title(title):
    encoded = tokenizer(
    title.numpy().decode('utf-8'),
    padding='max_length',
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="tf"
    )
    return encoded["input_ids"][0], encoded["attention_mask"][0]

def tokenize_title_tf(title):
# Wrap for TF mapping
    input_ids, attention_mask = tf.py_function(tokenize_title, [title], [tf.int32, tf.int32])
    input_ids.set_shape([MAX_LEN])
    attention_mask.set_shape([MAX_LEN])
    return input_ids, attention_mask

# -------------------------------------------------------

# Bucketize views safely

# -------------------------------------------------------
def bucketize_view_count(views):
    views = int(views)
    if views < 8000:
        return 0
    elif views < 50000:
        return 1
    else:
        return 2
    
# def bucketize_view_count(views):
#     views = int(float(views))
#     if views < 1000:
#         return 0
#     elif views < 5000:
#         return 1
#     elif views < 20000:
#         return 2
#     elif views < 100000:
#         return 3
#     else:
#         return 4
# -------------------------------------------------------

# Dataset loader

# -------------------------------------------------------
def preprocess(img_path, title, label):
    img = load_and_process_image("data/" + img_path)
    input_ids, attention_mask = tokenize_title_tf(title)
    features = {
        "image_input": img,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    return features, label

def split_dataset_stratified(csv_path: str,
                             train_frac: float = 0.8,
                             batch_size: int = 32,
                             random_state: int = 42)-> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build train/test datasets using a stratified split on labels.
    This recreates the tf.data pipelines from the CSV lists using the preprocess function
    from your dataloader module so we get the exact same preprocessing (image resize, tokenization, etc.).
    Then we select only the image+label for training the image model.
    """
    from sklearn.model_selection import train_test_split

    # Read CSV to get arrays (the same columns your dataloader expects)
    df = pd.read_csv(csv_path,
                     header=0,
                     names=["views", "image_path", "title","bucket"],
                     quotechar='"',
                     escapechar='\\',
                     engine="python")

    # Bucketize labels the same way your dataloader does
    # df["bucket"] = df["views"].apply(bucketize_view_count).astype("int32")
    df["bucket"] = df["bucket"].astype("int32")
    image_paths = df["image_path"].tolist()
    titles = df["title"].tolist()
    labels = df["bucket"].tolist()

    # Get indices and do stratified split
    indices = list(range(len(labels)))
    train_idx, test_idx = train_test_split(indices, train_size=train_frac, stratify=labels, random_state=random_state)

    # Build tf.data.Dataset from selected subsets
    # We will reuse dataload.preprocess which expects (img_path, title, label)
    def make_ds_from_indices(idxs):
        selected_image_paths = [image_paths[i] for i in idxs]
        selected_titles = [titles[i] for i in idxs]
        selected_labels = [labels[i] for i in idxs]
        ds = tf.data.Dataset.from_tensor_slices((selected_image_paths, selected_titles, selected_labels))
        # Map using the same preprocess function so images are processed identically
        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)  # yields (features_dict, label)
        ds = ds.map(lambda features, label: (features["image_input"], label),
                    num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = make_ds_from_indices(train_idx)
    test_ds = make_ds_from_indices(test_idx)
    return train_ds, test_ds

def split_text_dataset_stratified(csv_path: str,
                                  train_frac: float = 0.8,
                                  batch_size: int = 32,
                                  random_state: int = 42) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Returns TF datasets that yield ({"input_ids": ..., "attention_mask": ...}, label).
    Uses the same preprocessing tokenize_title_tf so tokenization is consistent.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path,
                     header=0,
                     names=["views", "image_path", "title","bucket"],
                     quotechar='"',
                     escapechar='\\',
                     engine="python")

    df["bucket"] = df["bucket"].astype("int32")
    image_paths = df["image_path"].tolist()
    titles = df["title"].tolist()
    labels = df["bucket"].tolist()

    indices = list(range(len(labels)))
    train_idx, test_idx = train_test_split(indices, train_size=train_frac, stratify=labels, random_state=random_state)

    def make_ds_from_indices(idxs):
        selected_image_paths = [image_paths[i] for i in idxs]  # image paths kept, but will not be used here
        selected_titles = [titles[i] for i in idxs]
        selected_labels = [labels[i] for i in idxs]
        ds = tf.data.Dataset.from_tensor_slices((selected_titles, selected_labels))
        # titles are strings, labels are ints
        # map to tokenized features
        def map_title_to_features(title, label):
            input_ids, attention_mask = tokenize_title_tf(title)
            features = {"input_ids": input_ids, "attention_mask": attention_mask}
            return features, label
        ds = ds.map(map_title_to_features, num_parallel_calls=AUTOTUNE)  # yields (features_dict, label)
        # Transform to (feature_inputs, label) where feature_inputs is a dict acceptable by model.fit
        ds = ds.map(lambda features, label: ({"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}, label),
                    num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = make_ds_from_indices(train_idx)
    test_ds = make_ds_from_indices(test_idx)
    return train_ds, test_ds

if __name__ == "__main__":
    train_ds, test_ds = split_text_dataset_stratified(
                            csv_path="data/train_split.csv",
                            train_frac=0.8,
                            batch_size=32,
                            random_state=42)
    print("Train dataset:", train_ds)

