import tensorflow as tf
import numpy as np
from data_load import split_text_dataset_stratified, MAX_LEN, AUTOTUNE
from transformers import TFBertModel
import os
from plot import plot_metrics

CSV_PATH = "data/train_split.csv"
BATCH_SIZE = 32
EPOCHS = 5
NUM_LABELS = 2  # buckets 
TEXT_HEAD_WEIGHTS = "text_head_weights.npz"  # file that will contain only the dense-head weights


def build_text_model(num_labels=NUM_LABELS, max_len=MAX_LEN, bert_name="bert-base-chinese"):
    # Load a TensorFlow BERT encoder (no classification head)
    bert = TFBertModel.from_pretrained(bert_name)
    bert.trainable = False  # Freeze BERT layers
    # Build the text classification model
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    bert_outputs = bert(input_ids, attention_mask=attention_mask)
    pooled = bert_outputs.pooler_output  # (batch, hidden_size)

    # Name the dense layers to restore their weights
    x = tf.keras.layers.Dense(256, activation="relu", name="text_dense")(pooled)
    x = tf.keras.layers.Dropout(0.3)(x)

    # set embedding layer
    text_embedding = tf.keras.layers.Dense(128, activation=None, name="text_embedding")(x)
    output= tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(text_embedding)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output, name="text_classifier")
    return model

def save_text_head_weights(path: str, model: tf.keras.Model):
    """
    Save only the dense-head weights (the two Dense layers before classifier that produce the embedding).
    We save:
      - text_dense kernel + bias
      - text_embedding kernel + bias
    as a compressed npz file so fusion_model can reload them into a rebuilt text embedding.
    """
    w_dense = model.get_layer("text_dense").get_weights()  # [kernel, bias]
    w_emb = model.get_layer("text_embedding").get_weights()  # [kernel, bias]

    np.savez_compressed(path,
                        text_dense_w=w_dense[0], text_dense_b=w_dense[1],
                        text_embedding_w=w_emb[0], text_embedding_b=w_emb[1])
    print(f"Saved text head weights to {path}")


def main():
    print("Building text-only datasets from CSV...")
    train_ds, test_ds = split_text_dataset_stratified(CSV_PATH, train_frac=0.9, batch_size=BATCH_SIZE, random_state=42)

    # For training performance
    train_ds = train_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # Build model
    model = build_text_model()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    model.summary()

    # Train
    history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
    plot_metrics(history) 

    # Export embedding-only model (recreate a model that outputs the embedding vector)
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("text_embedding").output, name="text_embedding_model")
    embedding_path = "text_embedding_model.keras"
    embedding_model.save(embedding_path)
    print(f"Saved embedding model to {embedding_path}")

    # Save only dense-head weights so fusion_model can reload into a rebuilt transformer-based embedding model
    save_text_head_weights(TEXT_HEAD_WEIGHTS, model)


if __name__ == "__main__":
    main()