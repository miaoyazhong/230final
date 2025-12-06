import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
from tensorflow.keras import layers, Model
import pandas as pd
# Import data loader
from data_load import split_dataset_stratified, IMG_SIZE, AUTOTUNE
from plot import plot_metrics

CSV_PATH = "data/train_split.csv"
BATCH_SIZE = 32

def print_element_spec(ds, name="dataset"):
    print(f"\n{name} element_spec:")
    print(ds.element_spec)

def count_samples(ds):
    # ds yields (images, labels) batches; this sums batch sizes to get total samples
    total = ds.reduce(tf.constant(0), lambda acc, batch: acc + tf.shape(batch[0])[0])
    return int(total.numpy())

def label_distribution_from_df(csv_path):
    df = pd.read_csv(csv_path,
                     header=0,
                     names=["views", "image_path", "title","bucket"],
                     quotechar='"',
                     escapechar='\\',
                     engine="python")
   
    return dict(df["bucket"].value_counts().sort_index())

def label_distribution_from_dataset(ds):
    # Unbatch for per-sample iteration and count labels
    c = Counter()
    for _, label in ds.unbatch().as_numpy_iterator():
        c[int(label)] += 1
    # Ensure keys 0..4 exist even if zero
    return {i: c.get(i, 0) for i in range(3)}

def show_one_batch_images(ds, n=9):
    for images, labels in ds.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        print("Batch images shape:", images_np.shape)
        print("Batch labels shape:", labels_np.shape, "labels sample:", labels_np[:min(10, len(labels_np))])
        # Plot the first n images
        n = min(n, images_np.shape[0])
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axs = axs.flatten()
        for i in range(n):
            axs[i].imshow(images_np[i])
            axs[i].set_title(f"label={labels_np[i]}")
            axs[i].axis("off")
        for j in range(n, len(axs)):
            axs[j].axis("off")
        plt.tight_layout()
        plt.show()
        break

def bili_model_with_classifier(image_shape=IMG_SIZE, embedding_dim=128, num_classes=3):
    input_shape = image_shape + (3,)
    # Load a MobileNetV2 base; replace weights path or set weights='imagenet' to use TF Hub weights
    base_model_path = "imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=base_model_path,
    )
    base_model.trainable = False

    base_model._name = "mobilenet_base"
    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    
    # preprocess_input for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # include dropout with probability of 0.2 to avoid overfitting
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    # # fully connected layer with ReLU activation
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    # Embedding output (this is what we will extract later)
    feature_output = tf.keras.layers.Dense(embedding_dim, activation=None, name="image_embedding")(x)
    # use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
 
    # Model returns both embedding and classifier outputs
    model = Model(inputs=inputs, outputs=outputs, name="image_model_with_classifier")

    return model

def unfreeze_backbone_for_finetune(model, base_model_name="mobilenet_base", unfreeze_last_n=30, freeze_bn=True):
    """
    Unfreezes the last `unfreeze_last_n` layers of the backbone (base model).
    - model: the full Keras model returned by bili_model_with_classifier
    - base_model_name: the name assigned to the MobileNetV2 base when building the model
    - unfreeze_last_n: how many layers from the end of the backbone to set trainable=True
    - freeze_bn: keep BatchNormalization layers frozen (recommended)
    """
    try:
        base_model = model.get_layer(base_model_name)
    except ValueError:
        raise ValueError(f"Base model with name '{base_model_name}' not found in model. Available layers: {[l.name for l in model.layers]}")

    # Set all layers non-trainable first
    for layer in base_model.layers:
        layer.trainable = False

    if unfreeze_last_n <= 0:
        return

    unfreeze_at = max(0, len(base_model.layers) - unfreeze_last_n)
    for layer in base_model.layers[unfreeze_at:]:
        layer.trainable = True

    if freeze_bn:
        # It's common to keep BatchNormalization layers frozen when fine-tuning
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

def main():
    print("CSV label distribution (raw):")
    print(label_distribution_from_df(CSV_PATH))

    train_ds, test_ds = split_dataset_stratified(csv_path=CSV_PATH,
                                                 train_frac=0.9,
                                                 batch_size=BATCH_SIZE,
                                                 random_state=42)

    print_element_spec(train_ds, "train_ds")
    print_element_spec(test_ds, "test_ds")

    train_count = count_samples(train_ds)
    test_count = count_samples(test_ds)
    print(f"\nTrain samples: {train_count}")
    print(f"Test samples: {test_count}")
    print(f"Total from split: {train_count + test_count}")

    print("\nLabel distribution in train dataset:")
    print(label_distribution_from_dataset(train_ds))
    print("\nLabel distribution in test dataset:")
    print(label_distribution_from_dataset(test_ds))

    print("\nInspect one batch from train_ds (shapes, dtypes, images):")
    show_one_batch_images(train_ds, n=9)

    num_classes = 3
    # Build the model with the backbone frozen initially
    model = bili_model_with_classifier(IMG_SIZE, embedding_dim=128, num_classes=num_classes)

    base_learning_rate = 0.001
    losses = tf.keras.losses.BinaryCrossentropy()
    metrics=['accuracy']
    # Keras needs a loss for every output; we only have the classifier output
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=base_learning_rate),
                loss=losses,
                metrics=metrics) 
    
    model.summary()
    # Note: your train_ds is already batched. It's good to shuffle and prefetch for training:
    train_for_fit = train_ds.prefetch(buffer_size=AUTOTUNE)
    test_for_fit = test_ds.prefetch(buffer_size=AUTOTUNE)

    # run initial training (transfer-learning)
    initial_epochs = 5
    print("Running initial training (transfer learning)...")
    history = model.fit(train_for_fit, validation_data=test_for_fit, epochs=initial_epochs)
    plot_metrics(history, None) 

    # Fine-tune: unfreeze the top layers of the backbone, recompile with a lower LR, and train
    print("Preparing to fine-tune the model...")

    # 1) Unfreeze the last N layers of the backbone. Tune this number for your task.
    unfreeze_last_n = 20  # try 20..50 depending on backbone depth and dataset size

    # Use our helper to unfreeze
    unfreeze_backbone_for_finetune(model, base_model_name="mobilenet_base", unfreeze_last_n=unfreeze_last_n, freeze_bn=True)

    # 2) Recompile with a much smaller learning rate for fine-tuning
    ft_learning_rate = 1e-5
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ft_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    # 3) Continue training (fine-tuning)
    fine_tune_epochs = 5
    total_epochs = initial_epochs + fine_tune_epochs
    print(f"Fine-tuning for {fine_tune_epochs} epochs (total epochs will be {total_epochs})...")
    history2 = model.fit(train_for_fit, validation_data=test_for_fit, epochs=fine_tune_epochs)
    plot_metrics(history2, None) 

    # After training, extract the embedding-only model
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("image_embedding").output, name="image_embedding_model")
    # Save both the full model and embedding model
    # model.save("image_model_with_classifier.keras")
    embedding_model.save("image_embedding_model.keras")
    print("Saved image_embedding_model.keras")

    import numpy as np
    y_true = []
    y_pred = []
    for x, y in test_ds.unbatch().batch(256):
        p = model.predict(x)
        y_true.append(y.numpy())
        y_pred.append(np.argmax(p, axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()