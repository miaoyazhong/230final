import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
from plot import plot_metrics


BATCH_SIZE = 32
IMG_SIZE = (200, 320)
# Load datasets and split into training and validation sets
directory = "dataset/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.1,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.1,
                                             subset='validation',
                                             seed=42)

# Get class names
class_names = train_dataset.class_names
print("Class names:", class_names)

def expand_labels(image, label):
    return image, tf.expand_dims(label, axis=-1)
# Expand label dimensions to use f1_score metric
train_dataset = train_dataset.map(expand_labels)
validation_dataset = validation_dataset.map(expand_labels)

# Optimize dataset performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# Define the preprocess input function from MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def bili_model(image_shape=IMG_SIZE):
    ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
    Returns:
        tf.keras.model
    '''

    input_shape = image_shape + (3,)
    base_model_path="imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"

    # Load the MobileNetV2 model, pre-trained on ImageNet, without the top classification layer
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,  
                                                   weights=base_model_path)
    
    # freeze the base model by making it non trainable
    base_model.trainable = False
    base_model._name = "mobilenet_base"
    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    
    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(inputs) 
    x = base_model(x, training=False) 

    # use global avg pooling to summarize the info in each channel
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # include dropout and batch normalization to avoid overfitting
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # # fully connected layer with ReLU activation
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    #embedding layer
    feature_output = tf.keras.layers.Dense(128, activation=None, name="image_embedding")(x)
    # use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(feature_output)
 
    model = tf.keras.Model(inputs, outputs)
    
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
        #keep BatchNormalization layers frozen when fine-tuning
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


# Create and compile the model
model = bili_model(IMG_SIZE)
base_learning_rate = 0.001

metrics=['accuracy']
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=metrics)

# Train the model
initial_epochs = 10
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

#Plot training results
plot_metrics(history) 

# Fine-tuning the model
# Unfreeze the last 20 layers of the backbone.
unfreeze_last_n = 20  
# Use our helper to unfreeze
unfreeze_backbone_for_finetune(model, base_model_name="mobilenet_base", unfreeze_last_n=unfreeze_last_n, freeze_bn=True)

# Recompile with a much smaller learning rate for fine-tuning
ft_learning_rate = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ft_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

# Continue training (fine-tuning)
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
print(f"Fine-tuning for {fine_tune_epochs} epochs (total epochs will be {total_epochs})...")
history2 = model.fit(train_dataset, validation_data=validation_dataset,epochs=fine_tune_epochs)
plot_metrics(history2) 

# After training, extract the embedding-only model
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("image_embedding").output, name="image_embedding_model")
# Save both the embedding model
embedding_model.save("image_embedding_model.keras")
print("Saved image_embedding_model.keras")