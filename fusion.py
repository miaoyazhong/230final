import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Input
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import TFBertModel
from data_load import preprocess, IMG_SIZE, MAX_LEN, AUTOTUNE

CSV_PATH = "data/train_split.csv"
BATCH_SIZE = 32
EPOCHS = 3
IMAGE_EMBED_PATH = "image_embedding_model.keras"
TEXT_EMBED_PATH = "text_embedding_model.keras"
TEXT_HEAD_WEIGHTS = "text_head_weights.npz" 

# Create multimodal datasets from CSV with stratified split
def make_multimodal_datasets(csv_path: str,train_frac: float = 0.9,batch_size: int = 32, random_state: int = 42):
    df = pd.read_csv(csv_path,
                     header=0,
                     names=["views", "image_path", "title", "bucket"],
                     quotechar='"',
                     escapechar='\\',
                     engine="python")
    df["bucket"] = df["bucket"].astype("int32")
    image_paths = df["image_path"].tolist()
    titles = df["title"].tolist()
    labels = df["bucket"].tolist()

    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(indices, train_size=train_frac, stratify=labels, random_state=random_state)

    def ds_from_indices(idxs):
        sel_image_paths = [image_paths[i] for i in idxs]
        sel_titles = [titles[i] for i in idxs]
        sel_labels = [labels[i] for i in idxs]
        ds = tf.data.Dataset.from_tensor_slices((sel_image_paths, sel_titles, sel_labels))
        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)  # yields (features_dict, label)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = ds_from_indices(train_idx)
    val_ds = ds_from_indices(val_idx)
    return train_ds, val_ds

# Build the fusion model by combining image and text embeddings
def build_fusion_model(image_path=IMAGE_EMBED_PATH,
                       text_weights=TEXT_HEAD_WEIGHTS,
                       image_trainable=False,
                       text_trainable=False,
                       ):
    # Image branch
    image_input = Input(shape=IMG_SIZE + (3,), name="image_input")
    input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")
    
    image_embedding_model = load_model(image_path, compile=False)
    image_embedding_model.trainable = image_trainable
    img_emb = image_embedding_model(image_input)
    print(f"Loaded image embedding model from {image_path}")

    # Text branc
    bert = TFBertModel.from_pretrained("bert-base-chinese")
    bert.trainable = text_trainable
    bert_out = bert(input_ids, attention_mask=attention_mask)
    pooled = bert_out.pooler_output
    # Use the same layer names as in text_model.py so we can load head weights
    x = tf.keras.layers.Dense(256, activation="relu", name="text_dense")(pooled)
    x = tf.keras.layers.Dropout(0.3)(x)
    txt_emb = tf.keras.layers.Dense(128, activation=None, name="text_embedding")(x)
    text_embedding_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=txt_emb, name="text_embedding_model_rebuilt")
    # load head weights
    npz = np.load(text_weights)
    w_td = npz["text_dense_w"]
    b_td = npz["text_dense_b"]
    w_te = npz["text_embedding_w"]
    b_te = npz["text_embedding_b"]
    text_embedding_model.get_layer("text_dense").set_weights([w_td, b_td])
    text_embedding_model.get_layer("text_embedding").set_weights([w_te, b_te])
    print(f"Loaded text head weights from {text_weights} into rebuilt text embedding model.")

    # Fusion and classification head
    fusion = Concatenate()([img_emb, txt_emb])
    x = Dense(256, activation="relu")(fusion)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid", name="virality_output")(x)

    model = Model(inputs=[image_input, input_ids, attention_mask], outputs=output, name="multimodal_fusion_model")
    return model


def main(csv_path=CSV_PATH, batch_size=BATCH_SIZE, epochs=EPOCHS, train_frac=0.9, fine_tune_image=False,         fine_tune_text=False):
    print("Building datasets...")
    train_ds, val_ds = make_multimodal_datasets(csv_path, train_frac=train_frac, batch_size=batch_size)
    print("Attempting to load pretrained embedding models...")
    

    print("Building fusion model...")
    fuse_model = build_fusion_model(IMAGE_EMBED_PATH,
                                    TEXT_HEAD_WEIGHTS,
                                    image_trainable=fine_tune_image,
                                    text_trainable=fine_tune_text)
    # Change learning rate if fine-tuning
    if (fine_tune_image or fine_tune_text):
        learn_rate = 0.0001
    else:
        learn_rate = 0.001

    # Compile model
    fuse_model.compile(optimizer=tf.keras.optimizers.Adam(learn_rate),
                       loss="BinaryCrossentropy",
                       metrics=["accuracy"])

    fuse_model.summary()

    train_for_fit = train_ds.prefetch(AUTOTUNE)
    val_for_fit = val_ds.prefetch(AUTOTUNE)

    print("Starting training...")
    history = fuse_model.fit(train_for_fit, validation_data=val_for_fit, epochs=epochs)

    save_path = "multimodal_virality_model.keras"
    fuse_model.save(save_path)
    print(f"Saved multimodal model to {save_path}")

    return fuse_model, history


if __name__ == "__main__":
    model, history = main(epochs=EPOCHS , fine_tune_image=False, fine_tune_text=False)
   
 