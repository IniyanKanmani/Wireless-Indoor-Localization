import io
import os
import zipfile
import datetime
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn as sklearn
import matplotlib.pyplot as plt




def unzip_file(filename,
               dest_path=''):

    """
    Unzip Zipfile.
    """
    
    zip_ref = zipfile.ZipFile(os.path.join(filename))
    zip_ref.extractall(os.path.join(dest_path))
    zip_ref.close()




def walk_through_dir(dir_path):

    """
    Walks through dir_path returning its contents.
    """

    for dirpath, dirnames, filenames in os.walk(os.path.join(dir_path)):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")




def lr_scheduler(epoch,
                 lr):

    """
    Create a Learning Rate Scheduler.
    """

    return 1e-8 * 10 ** (epoch / 20)




def create_early_stoppping_callback(monitor='val_loss',
                                    min_delta=0.02,
                                    patience=5,
                                    verbose=0,
                                    mode='min',
                                    baseline=None):

    """
    Stop Training Early.
    """                                    

    return tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                            min_delta=min_delta,
                                            patience=patience,
                                            verbose=verbose,
                                            mode=mode,
                                            baseline=baseline)




def create_model_checkpoint_callback(dir_name,
                                     model_name,
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min',
                                     save_freq='epoch'):

    """
    Save model using Model Checkpoint.
    """

    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(dir_name, model_name, 'checkpoint'),
                                              monitor=monitor,
                                              verbose=verbose,      
                                              save_best_only=save_best_only,
                                              save_weights_only=save_weights_only,
                                              mode=mode,
                                              save_freq=save_freq)




def create_tensorboard_callback(dir_name,
                                model_name):

    """
    Save TensorBoard Log Files.
    """

    log_dir = os.path.join(dir_name, model_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    print(f'Saving TensorBoard log files to: {log_dir}')

    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)




def history_to_df(history):

    """
    Convert History object into a DataFrame.
    """

    return pd.DataFrame(history.history)




def plot_model(model,
               dir_name,
               model_name,
               show_shapes=True,
               show_dtype=False,
               show_layer_names=True,
               show_layer_activations=False):

    """
    Plot and save the Model.
    """

    file_path = os.path.join(dir_name, model_name)

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    to_file = os.path.join(file_path, 'model.png')

    return tf.keras.utils.plot_model(model=model,
                                     to_file=to_file,
                                     show_shapes=show_shapes,
                                     show_dtype=show_dtype,
                                     show_layer_names=show_layer_names,
                                     show_layer_activations=show_layer_activations)




def plot_history(history,
                 metrics=['loss', 'accuracy'],
                 initial_index=0,
                 final_index=None,
                 training=True,
                 validation=True,
                 learning_rate=False,
                 save_fig=False,
                 fig_path=''):

    """
    Plots history graphs for the given History object.
    """

    nrows = len(metrics) + (1 if learning_rate else 0)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(14, nrows * 8))

    fig.suptitle("History Plot")

    epochs = np.array(history.epoch) + 1

    if len(metrics) == 1:
        
        if training:
            train = history.history[metrics[0]]
            ax.plot(epochs[initial_index : final_index], train[initial_index : final_index], c='r', label=f'Training {str(metrics[0]).capitalize()}', linestyle='--')

        if validation:
            val = history.history['val_' + metrics[0]]
            ax.plot(epochs[initial_index : final_index], val[initial_index : final_index], c='b', label=f'Validation {str(metrics[0]).capitalize()}', linestyle='-.')
        
        ax.set_xlabel('Number of Epochs')
        ax.set_ylabel(str(metrics[0]).capitalize())
        ax.legend();

    elif learning_rate and nrows == 1:

        lr = history.history['lr']

        if training:
            train = history.history['loss']
            ax.semilogx(lr[initial_index : final_index], train[initial_index : final_index], c='r', label=f'Training Loss', linestyle='--')

        if validation:
            val = history.history['val_loss']
            ax.semilogx(lr[initial_index : final_index], val[initial_index : final_index], c='b', label=f'Validation Loss', linestyle='-.')
        
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.legend();
    
    elif len(metrics) > 1:
        
        for i, metric in enumerate(metrics):

            if training:
                train = history.history[metric]
                ax[i].plot(epochs[initial_index : final_index], train[initial_index : final_index], c='r', label=f'Training {str(metric).capitalize()}', linestyle='--')

            if validation:
                val = history.history[('val_' if metric != 'lr' else '') + metric]
                ax[i].plot(epochs[initial_index : final_index], val[initial_index : final_index], c='b', label=f'Validation {str(metric).capitalize()}', linestyle='-.')
            
            ax[i].set_xlabel('Number of Epochs')
            ax[i].set_ylabel(str(metric).capitalize())
            ax[i].legend();
        
        if learning_rate:

            lr = history.history['lr']

            if training:
                train = history.history['loss']
                ax[nrows - 1].semilogx(lr[initial_index : final_index], train[initial_index : final_index], c='r', label=f'Training Loss', linestyle='--')

            if validation:
                val = history.history['val_loss']
                ax[nrows - 1].semilogx(lr[initial_index : final_index], val[initial_index : final_index], c='b', label=f'Validation Loss', linestyle='-.')

            ax[nrows - 1].set_xlabel('Learning Rate')
            ax[nrows - 1].set_ylabel('Loss')
            ax[nrows - 1].legend();

    if save_fig:
        fig.savefig(fig_path)




def plot_combined_history(historys,
                          metrics=['loss', 'accuracy'],
                          initial_index=0,
                          final_index=None,
                          training=True,
                          validation=True,
                          save_fig=False,
                          fig_path=''):

    """
    Plots combined history graphs for the given History object List.
    """

    nrows = len(metrics)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(14, nrows * 8))

    fig.suptitle("Combined History Plot")

    epochs = []
    for history in historys:
        epochs.append(history.epoch)

    ep = np.array(epochs).reshape(1, -1).squeeze() + 1
    epochs = np.array(epochs) + 1

    if len(metrics) == 1:
        
        train = []
        if training:

            for history in historys:
                train.append(history.history[metrics[0]])

            train = np.array(train).reshape(1, -1).squeeze()

            ax.plot(ep[initial_index : final_index], train[initial_index : final_index], c='r', label=f'Training {str(metrics[0]).capitalize()}', linestyle='--')

        val = []
        if validation:

            for history in historys:
                val.append(history.history['val_' + metrics[0]])

            val = np.array(val).reshape(1, -1).squeeze()

            ax.plot(ep[initial_index : final_index], val[initial_index : final_index], c='b', label=f'Validation {str(metrics[0]).capitalize()}', linestyle='-.')

        min = 0
        max = 0

        if training and validation:
            min = np.min(train[initial_index : final_index]) if np.min(train[initial_index : final_index]) < np.min(val[initial_index : final_index]) else np.min(val[initial_index : final_index])
            max = np.max(train[initial_index : final_index]) if np.max(train[initial_index : final_index]) > np.max(val[initial_index : final_index]) else np.max(val[initial_index : final_index])
        elif training:
            min = np.min(train[initial_index : final_index])
            max = np.max(train[initial_index : final_index])
        elif validation:
            min = np.min(val[initial_index : final_index])
            max = np.max(val[initial_index : final_index])

        for e in epochs[:1]:
            if e[-1] > initial_index and e[-1] < final_index:
                ax.plot([e[-1], e[-1]], [min, max], c='g', linestyle=':')
        
        ax.set_xlabel('Number of Epochs')
        ax.set_ylabel(str(metrics[0]).capitalize())
        ax.legend();
    
    elif len(metrics) > 1:
        
        for i, metric in enumerate(metrics):

            train = []
            if training:

                for history in historys:
                    train.append(history.history[metric])

                train = np.array(train).reshape(1, -1).squeeze()

                ax[i].plot(ep[initial_index : final_index], train[initial_index : final_index], c='r', label=f'Training {str(metrics[0]).capitalize()}', linestyle='--')

            val = []
            if validation:

                for history in historys:
                    val.append(history.history['val_' + metric])

                val = np.array(val).reshape(1, -1).squeeze()

                ax[i].plot(ep[initial_index : final_index], val[initial_index : final_index], c='b', label=f'Validation {str(metrics[0]).capitalize()}', linestyle='-.')

            min = 0
            max = 0

            if training and validation:
                min = np.min(train[initial_index : final_index]) if np.min(train[initial_index : final_index]) < np.min(val[initial_index : final_index]) else np.min(val[initial_index : final_index])
                max = np.max(train[initial_index : final_index]) if np.max(train[initial_index : final_index]) > np.max(val[initial_index : final_index]) else np.max(val[initial_index : final_index])
            elif training:
                min = np.min(train[initial_index : final_index])
                max = np.max(train[initial_index : final_index])
            elif validation:
                min = np.min(val[initial_index : final_index])
                max = np.max(val[initial_index : final_index])

            for e in epochs[:1]:
                if e[-1] > initial_index and e[-1] < final_index:
                    ax[i].plot([e[-1], e[-1]], [min, max], c='g', linestyle=':')
            
            ax[i].set_xlabel('Number of Epochs')
            ax[i].set_ylabel(str(metrics[0]).capitalize())
            ax[i].legend();

    if save_fig:
        fig.savefig(fig_path)




def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          figsize=(10,10),
                          text_size=15,
                          save_fig=False,
                          fig_path=''):

    """
    Plot a Confusion Matrix for any number of classes.
    """

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
      labels = classes
    else:
      labels = np.arange(cm.shape[0])

    ax.set(title="Confusion Matrix",
          xlabel="Predicted Label",
          ylabel="True Label",
          xticks=np.arange(n_classes),
          yticks=np.arange(n_classes),
          xticklabels=labels, 
          yticklabels=labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom() 
    plt.xticks(rotation=90, fontsize=text_size)
    plt.yticks(fontsize=text_size)
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    threshold = (cm.max() + cm.min()) / 2.  
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
               horizontalalignment="center",
               color="white" if cm[i, j] > threshold else "black",
               size=text_size)

    if save_fig:
        fig.savefig(fig_path)




def calculate_classification_metrics(y_true,
                                     y_pred,
                                     average='binary',
                                     print_values=False):

    """
    Calculate accuracy, precision, recall, f1, confusion_matrix_score, classification_report_score for the given y_true, y_pred.
    """
    
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred, average=average)
    recall = sklearn.metrics.recall_score(y_true, y_pred, average=average)
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average=average)
    confusion_matrix_score = sklearn.metrics.confusion_matrix(y_true, y_pred)
    classification_report_score = sklearn.metrics.classification_report(y_true, y_pred)

    if print_values:
        print('Accuracy:               ' + str(accuracy))
        print('Precision:              ' + str(precision))
        print('Recall:                 ' + str(recall))
        print('F1                      ' + str(f1))
        print('Confusion Matrix:      \n' + str(confusion_matrix_score))
        print('Classification Report: \n' + str(classification_report_score))

    return {'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'f1' : f1,
            'confusion_matrix' : confusion_matrix_score,
            'classification_report' : classification_report_score}




def calculate_regression_metrics(y_true,
                                 y_pred,
                                 dtype=np.float32,
                                 print_values=False):

    """
    Calculate mae, mse, rmse, mape, mase for the given y_true, y_pred.
    """

    y_true = np.array(y_true, dtype=dtype)
    y_pred = np.array(y_pred, dtype=dtype)

    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mse / sklearn.metrics.mean_absolute_error(y_true[1:], y_true[:-1])

    if print_values:
        print('MAE:  ' + str(mae))
        print('MSE:  ' + str(mse))
        print('RMSE: ' + str(rmse))
        print('MAPE: ' + str(mape))
        print('MASE: ' + str(mase))

    return {"mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "mase": mase}




def save_embeddings(dir_name,
                    experiment_name,
                    words_in_vocab,
                    embed_weights,
                    vector_name='vectors.tsv',
                    metadata_name='metadata.tsv'):
    
    """
    Save Embedings to vectors.tsv and metadata.tsv.
    """

    out_v = io.open(os.path.join(dir_name, experiment_name, vector_name), 'w', encoding='utf-8')
    out_m = io.open(os.path.join(dir_name, experiment_name, metadata_name), 'w', encoding='utf-8')

    for index, word in enumerate(words_in_vocab):
        if index == 0:
            continue

        vec = embed_weights[index]

        out_v.write('\t'.join([str(x) for x in vec]) + '\n')
        out_m.write(word + '\n')
    
    out_m.close()
    out_v.close()




def save_model(model,
               dir_name,
               model_name,
               hdf5=True):

    """
    Save model in SavedModel / HDF5 format.
    """

    path = os.path.join('models/' + dir_name + model_name + ('.h5' if hdf5 else ''))
    model.save(path)
    
    return path




def load_saved_model(dir_name,
                     model_name,
                     model_checkpoint=False):

    """
    Load SavedModel / HDF5 model.
    """
    
    path = os.path.join(dir_name, model_name, 'checkpoint/' if model_checkpoint else '')
    
    return tf.keras.models.load_model(path)
