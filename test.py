import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Reuse preprocessing helpers/constants from your dataloader
from data_load import preprocess, AUTOTUNE


# Configuration
TEST_CSV = "data/test_split.csv"
BATCH_SIZE = 32
MODEL_PATH = "multimodal_virality_model.keras"
NUM_LABELS = 3  # buckets 0..4


def make_test_dataset(csv_path: str, batch_size: int = BATCH_SIZE):
    """
    Build a tf.data.Dataset from csv that yields (features_dict, label) compatible with the fusion model.
    Uses data_load.preprocess to ensure identical preprocessing.
    """
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

    ds = tf.data.Dataset.from_tensor_slices((image_paths, titles, labels))
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)  # yields (features_dict, label)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, df  # return df for optional diagnostic info (paths / titles)


def try_load_model(model_path: str):
    """
    Attempt to load the saved multimodal model. If it fails and the fusion helpers
    are available, rebuild the model architecture using fusion.build_fusion_model and
    (optionally) load weights-only.
    Returns a compiled model ready for evaluation.
    """
    model = load_model(model_path, compile=False)
    print(f"Loaded full model from {model_path}")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def evaluate_model_on_testset(model, test_ds):
    """
    Evaluate using model.evaluate (gives loss/accuracy) and also compute
    per-class precision/recall/f1 and confusion matrix using sklearn.
    """
    print("Running model.evaluate on test dataset (loss / accuracy):")
    results = model.evaluate(test_ds, verbose=1)
    # results may be [loss, accuracy] depending on compiled metrics
    print("Evaluate results:", results)

    # Predict and compute classification metrics
    print("Running model.predict to collect per-sample predictions...")
    preds = model.predict(test_ds, verbose=1)  # shape (N, num_labels)
    y_pred = np.argmax(preds, axis=1)

    # Collect true labels from dataset
    y_true_parts = []
    for _, batch_labels in test_ds:
        y_true_parts.append(batch_labels.numpy())
    if len(y_true_parts) == 0:
        raise RuntimeError("Test dataset contained no samples.")
    y_true = np.concatenate(y_true_parts, axis=0)

    # Sanity: ensure matching lengths
    if len(y_true) != len(y_pred):
        # It's possible that predict returned fewer/more samples if dataset had partial batch handling differences.
        # Try to trim/pad to the minimum length to get metrics.
        n = min(len(y_true), len(y_pred))
        print(f"Warning: mismatch between y_true ({len(y_true)}) and y_pred ({len(y_pred)}). Truncating to {n}.")
        y_true = y_true[:n]
        y_pred = y_pred[:n]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}    Micro F1: {micro_f1:.4f}\n")

    print("Classification report (per-class precision / recall / f1):")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    return {
        "evaluate_results": results,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "classification_report": classification_report(y_true, y_pred, digits=4, output_dict=True),
        "confusion_matrix": cm
    }


def main():
    print("Preparing test dataset from:", TEST_CSV)
    test_ds, df = make_test_dataset(TEST_CSV, batch_size=BATCH_SIZE)
    print(f"Test samples: {len(df)}")

    print("Loading or rebuilding the model...")
    model = try_load_model(MODEL_PATH)

    # Run evaluation & metrics
    metrics = evaluate_model_on_testset(model, test_ds)

    # Optionally save a CSV with predictions alongside original rows for inspection

    print("Saving per-sample predictions to predictions_on_test.csv")
    preds = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    df_out = df.copy().reset_index(drop=True)
    # df order must match dataset order: make_test_dataset constructed df and dataset in the same order
    df_out["pred_bucket"] = y_pred
    df_out.to_csv("predictions_on_test.csv", index=False)
    print("Saved predictions_on_test.csv")

    print("Done.")


if __name__ == "__main__":
    main()