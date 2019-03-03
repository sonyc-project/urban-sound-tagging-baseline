import argparse
import csv
import datetime
import json
import gzip
import os
import numpy as np

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_score, recall_score

import sonyc_data
from sonyc_data import load_sonyc_data


## HELPERS

def get_sensor_id(file_id):
    """
    Get sensor id from a SONYC file ID

    Parameters
    ----------
    file_id

    Returns
    -------
    sensor_id

    """
    return file_id.rsplit(' ', 1)[-1].split('_')[0]


def load_embeddings(file_list, emb_dir):
    """
    Load saved embeddings from an embedding directory

    Parameters
    ----------
    file_list
    emb_dir

    Returns
    -------
    embeddings
    ignore_idxs

    """
    embeddings = []
    ignore_idxs = []
    for idx, filename in enumerate(file_list):
        emb_path = os.path.join(emb_dir, os.path.splitext(filename)[0] + '.npy.gz')
        try:
            with gzip.open(emb_path, 'rb') as f:
                embeddings.append(np.load(f))
        except:
            embeddings.append(None)
            ignore_idxs.append(idx)

    return embeddings, ignore_idxs


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


## METRICS

def binary_accuracy_round(y_true, y_pred):
    """
    Multi-label average accuracy, using 0.5 as the detection threshold. For use with Keras.

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------
    accuracy

    """
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)


def binary_accuracy_round_np(y_true, y_pred):
    """
    Multi-label average accuracy, using 0.5 as the detection threshold. For use with numpy.

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------
    accuracy

    """
    return np.mean(np.equal(np.round(y_true), np.round(y_pred)))


def classwise_binary_accuracy_round_np(y_true, y_pred):
    """
    Multi-label average accuracy per class, using 0.5 as the detection threshold. For use with numpy.

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------
    accuracy

    """
    return np.mean(np.equal(np.round(y_true), np.round(y_pred)), axis=0)


## MODEL CONSTRUCTION


def construct_mlp_framewise(emb_size, num_classes, hidden_layer_size=128, l2_reg=1e-5):
    """
    Construct a 2-hidden-layer MLP model for framewise processing

    Parameters
    ----------
    emb_size
    num_classes
    hidden_layer_size
    l2_reg

    Returns
    -------
    model

    """
    inp = Input(shape=(emb_size,), dtype='float32', name='input')
    y = Dense(hidden_layer_size, activation='relu',
              kernel_regularizer=regularizers.l2(l2_reg), name='dense1')(inp)
    y = Dense(hidden_layer_size,
              activation='relu', kernel_regularizer=regularizers.l2(l2_reg),
              name='dense2')(y)

    y = Dense(num_classes, activation='sigmoid',
              kernel_regularizer=regularizers.l2(l2_reg), name='output')(y)

    m = Model(inputs=inp, outputs=y)
    m.name = 'urban_sound_classifier'

    return m


## DATA PREPARATION

def prepare_framewise_data(train_file_idxs, test_file_idxs, embeddings, target_list, standardize=True):
    """
    Prepare inputs and targets for framewise training using training and evaluation indices.

    Parameters
    ----------
    train_file_idxs
    test_file_idxs
    embeddings
    target_list
    standardize

    Returns
    -------
    X_train
    y_train
    X_test
    y_test
    scaler

    """

    X_train = []
    y_train = []
    for idx in train_file_idxs:
        X_ = list(embeddings[idx])
        X_train += X_
        for _ in range(len(X_)):
            y_train.append(target_list[idx])

    train_idxs = np.random.permutation(len(X_train))

    X_train = np.array(X_train)[train_idxs]
    y_train = np.array(y_train)[train_idxs]

    X_test = []
    y_test = []
    for idx in test_file_idxs:
        X_ = list(embeddings[idx])
        X_test += X_
        for _ in range(len(X_)):
            y_test.append(target_list[idx])

    test_idxs = np.random.permutation(len(X_test))
    X_test = np.array(X_test)[test_idxs]
    y_test = np.array(y_test)[test_idxs]

    # standardize
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    return X_train, y_train, X_test, y_test, scaler


## GENERIC MODEL TRAINING

def train_mlp(model, x_train, y_train, output_dir, batch_size=64,
              num_epochs=100, patience=20, learning_rate=1e-4):
    """
    Train a MLP model with the given data.

    Parameters
    ----------
    model
    x_train
    y_train
    output_dir
    batch_size
    num_epochs
    patience
    learning_rate

    Returns
    -------
    history

    """

    loss = 'binary_crossentropy'
    metrics = [binary_accuracy_round]
    monitor = 'val_loss'
    #set_random_seed(random_state)

    os.makedirs(output_dir, exist_ok=True)

    # Set up callbacks
    cb = []
    # checkpoint
    model_weight_file = os.path.join(output_dir, 'model_best.h5')
    cb.append(keras.callbacks.ModelCheckpoint(model_weight_file,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor=monitor))
    # early stopping
    cb.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=patience))
    # monitor losses
    history_csv_file = os.path.join(output_dir, 'history.csv')
    cb.append(keras.callbacks.CSVLogger(history_csv_file, append=True,
                                        separator=','))

    # Fit model
    model.compile(Adam(lr=learning_rate), loss=loss, metrics=metrics)
    history = model.fit(
        x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
        callbacks=cb, verbose=2)

    return history


## MODEL TRAINING

def train_framewise(annotation_path, emb_dir, output_dir, exp_id, label_mode="low", batch_size=64,
                  num_epochs=100, patience=20, learning_rate=1e-4, hidden_layer_size=128,
                  l2_reg=1e-5, standardize=True, timestamp=None):
    """
    Train and evaluate a framewise MLP model.

    Parameters
    ----------
    annotation_path
    emb_dir
    output_dir
    exp_id
    label_mode
    batch_size
    test_ratio
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    l2_reg
    standardize
    timestamp

    Returns
    -------

    """
    file_list, high_target_list, low_target_list, train_file_idxs, test_file_idxs = load_sonyc_data(annotation_path)

    if label_mode == "low":
        target_list = low_target_list
        labels = sonyc_data.LOW_LEVEL_LABELS
    elif label_mode == "high":
        target_list = high_target_list
        labels = sonyc_data.HIGH_LEVEL_LABELS
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)

    embeddings, ignore_idxs = load_embeddings(file_list, emb_dir)

    # Remove files that couldn't be loaded
    train_file_idxs = [idx for idx in train_file_idxs if idx not in ignore_idxs]
    test_file_idxs = [idx for idx in test_file_idxs if idx not in ignore_idxs]

    X_train, y_train, X_test, y_test, scaler = prepare_framewise_data(train_file_idxs, test_file_idxs, embeddings,
                                                                        target_list, standardize=standardize)

    _, emb_size = X_train.shape

    model = construct_mlp_framewise(emb_size, num_classes, hidden_layer_size=hidden_layer_size, l2_reg=l2_reg)

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    results_dir = os.path.join(output_dir, exp_id, timestamp)

    history = train_mlp(model, X_train, y_train, results_dir, batch_size=batch_size,
              num_epochs=num_epochs, patience=patience, learning_rate=learning_rate)

    results = {}
    results['train'] = predict_framewise(embeddings, target_list, train_file_idxs, model, labels, scaler=scaler)
    results['test'] = predict_framewise(embeddings, target_list, test_file_idxs, model, labels, scaler=scaler)
    results['train_history'] = history.history

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    for aggregation_type, y_pred in results['test'].items():
        generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                             aggregation_type, label_mode)


## MODEL EVALUATION

def predict_framewise(embeddings, target_list, test_file_idxs, model, labels, scaler=None):
    """
    Evaluate the output of a framewise classification model.

    Parameters
    ----------
    embeddings
    target_list
    test_file_idxs
    model
    labels
    scaler

    Returns
    -------
    results
    """
    y_pred_max = []
    y_pred_mean = []
    y_pred_softmax = []

    for idx in test_file_idxs:
        if scaler is None:
            X_ = np.array(embeddings[idx])
        else:
            X_ = np.array(scaler.transform(embeddings[idx]))
        y_ = target_list[idx]
        pred_frame = model.predict(X_)

        y_pred_max.append(pred_frame.max(axis=0).tolist())
        y_pred_mean.append(pred_frame.mean(axis=0).tolist())
        y_pred_softmax.append(((softmax(pred_frame, axis=0) * pred_frame).sum(axis=0)).tolist())


    results = {
        'max': y_pred_max,
        'mean': y_pred_mean,
        'softmax': y_pred_softmax
    }

    return results


def generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                         aggregation_type, label_mode):
    output_path = os.path.join(results_dir, "output_{}.csv".format(aggregation_type))
    test_file_list = [file_list[idx] for idx in test_file_idxs]

    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + sonyc_data.LOW_LEVEL_LABELS + sonyc_data.HIGH_LEVEL_LABELS
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(test_file_list, y_pred):
            row = [filename]

            if label_mode == "low":
                # Add low level labels
                row += list(y)
                # Add placeholder values for high level
                row += [-1 for _ in range(len(sonyc_data.HIGH_LEVEL_LABELS))]

            else:
                # Add placeholder values for low level
                row += [-1 for _ in range(len(sonyc_data.LOW_LEVEL_LABELS))]
                # Add high level labels
                row += list(y)

            csvwriter.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("emb_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("exp_id", type=str)

    parser.add_argument("--hidden_layer_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no_standardize", action='store_true')
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--label_mode", type=str, choices=["low", "high"],
                        default='low')

    args = parser.parse_args()

    # save args to disk
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(args.output_dir, args.exp_id, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    kwarg_file = os.path.join(out_dir, "hyper_params.json")
    with open(kwarg_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_framewise(args.annotation_path,
                    args.emb_dir,
                    args.output_dir,
                    args.exp_id,
                    label_mode=args.label_mode,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    hidden_layer_size=args.hidden_layer_size,
                    l2_reg=args.l2_reg,
                    standardize=(not args.no_standardize),
                    timestamp=timestamp)
