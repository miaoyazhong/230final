import matplotlib.pyplot as plt

def plot_metrics(history, history_fine=None):
    ''' Plot training and validation accuracy/loss from training history
    Arguments:
        history -- tf.keras History object
    '''
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if history_fine != None:
        acc = [0.] + history_fine.history['accuracy']+ history.history['accuracy']
        val_acc = [0.] + history_fine.history['val_accuracy'] + history.history['val_accuracy']

        loss = history_fine.history['loss'] + history.history['loss']
        val_loss = history_fine.history['val_loss'] + history.history['val_loss']


    plt.figure(figsize=(8, 10)) 

    # ---- Row 1: Accuracy ----
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    # ---- Row 2: Loss ----
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Sparse Categorical Cross entropy')
    plt.ylim([0, 2.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    plt.show()          
