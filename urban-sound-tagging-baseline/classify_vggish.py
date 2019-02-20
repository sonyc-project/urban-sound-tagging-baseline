import argparse
import datetime
import json
import gzip
import os
import numpy as np

import keras
from keras.layers import Input, Dense, TimeDistributed
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
import keras.backend as K
from autopool import AutoPool1D
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_score, recall_score

from .sonyc_data import load_sonyc_data
from .balancing import mlsmote, lssmote
from . import sonyc_data


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

    """
    embeddings = []
    for filename in file_list:
        emb_path = os.path.join(emb_dir, os.path.splitext(filename)[0] + '.npy.gz')
        with gzip.open(emb_path, 'rb') as f:
            embeddings.append(np.load(f))

    return embeddings


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


def construct_mlp_mil(num_frames, emb_size, num_classes, hidden_layer_size=128, l2_reg=1e-5):
    """
    Construct a 2-hidden-layer MLP model for MIL processing

    Parameters
    ----------
    num_frames
    emb_size
    num_classes
    hidden_layer_size
    l2_reg

    Returns
    -------
    model

    """
    inp = Input(shape=(num_frames, emb_size), dtype='float32', name='input')
    y_frame = TimeDistributed(Dense(hidden_layer_size, activation='relu',
                                    kernel_regularizer=regularizers.l2(l2_reg), name='dense1'),
                              input_shape=(num_frames, emb_size))(inp)
    y_frame = TimeDistributed(Dense(hidden_layer_size, activation='relu',
                                    kernel_regularizer=regularizers.l2(l2_reg), name='dense2'),
                              input_shape=(num_frames, hidden_layer_size))(y_frame)

    y_label_frame = TimeDistributed(Dense(num_classes, activation='sigmoid',
                                          kernel_regularizer=regularizers.l2(l2_reg), name='output'),
                                    input_shape=(num_frames, hidden_layer_size))(y_frame)

    # Apply autopool over time dimension
    y = AutoPool1D(kernel_constraint=keras.constraints.non_neg(), axis=1)(y_label_frame)

    m = Model(inputs=inp, outputs=y)
    m.name = 'urban_sound_classifier'

    return m


## DATA PREPARATION

def get_val_test_file_split(file_list, eval_file_idxs, target_list, test_ratio=0.5):
    """
    Get train/validation/test split using iterative stratification, and optionally splitting by test sensors

    Parameters
    ----------
    file_list
    target_list
    test_ratio

    Returns
    -------
    valid_file_idxs
    test_file_idxs

    """
    assert 0 < test_ratio < 1

    val_ratio = 1.0 - test_ratio

    kfold = IterativeStratification(n_splits=2, sample_distribution_per_fold=[val_ratio, test_ratio])
    split_idxs = []
    for _, train_idxs in kfold.split(np.ones((len(eval_file_idxs), 13)),
                                     np.array([target_list[idx] for idx in eval_file_idxs])):
        split_idxs.append(np.array([eval_file_idxs[idx] for idx in train_idxs]))

    valid_file_idxs, test_file_idxs = split_idxs

    return valid_file_idxs, test_file_idxs


def prepare_framewise_data(train_file_idxs, valid_file_idxs, embeddings, target_list, standardize=True,
                           oversample=None, oversample_iters=1, thresh_type="mean"):
    """
    Prepare inputs and targets for framewise training using training and validation indices.

    Parameters
    ----------
    train_file_idxs
    valid_file_idxs
    embeddings
    target_list
    standardize
    oversample
    oversample_iters
    thresh_type

    Returns
    -------
    X_train
    y_train
    X_valid
    y_valid
    scaler

    """
    if oversample == 'mlsmote':
        X_train = np.array([embeddings[idx] for idx in train_file_idxs])
        y_train = np.array([target_list[idx] for idx in train_file_idxs])

        X_train_, y_train_, _ = mlsmote(X_train, y_train, oversample_iters=oversample_iters,
                                        thresh_type=thresh_type)

        X_train = []
        y_train = []
        for X, y in zip(X_train, y_train):
            X_train += list(X)
            y_train += [y for _ in range(len(X))]

        # Remove references
        X_train_ = None
        y_train_ = None
    elif oversample == 'lssmote':
        X_train = np.array([embeddings[idx] for idx in train_file_idxs])
        y_train = np.array([target_list[idx] for idx in train_file_idxs])

        X_train_, y_train_, _ = lssmote(X_train, y_train, oversample_iters=oversample_iters,
                                        thresh_type=thresh_type)

        X_train = []
        y_train = []
        for X, y in zip(X_train, y_train):
            X_train += list(X)
            y_train += [y for _ in range(len(X))]

        # Remove references
        X_train_ = None
        y_train_ = None
    elif oversample is None:

        X_train = []
        y_train = []
        for idx in train_file_idxs:
            X_ = list(embeddings[idx])
            X_train += X_
            for _ in range(len(X_)):
                y_train.append(target_list[idx])

    else:
        raise ValueError("Unknown oversample method: {}".format(oversample))

    train_idxs = np.random.permutation(len(X_train))

    X_train = np.array(X_train)[train_idxs]
    y_train = np.array(y_train)[train_idxs]

    X_valid = []
    y_valid = []
    for idx in valid_file_idxs:
        X_ = list(embeddings[idx])
        X_valid += X_
        for _ in range(len(X_)):
            y_valid.append(target_list[idx])

    valid_idxs = np.random.permutation(len(X_valid))
    X_valid = np.array(X_valid)[valid_idxs]
    y_valid = np.array(y_valid)[valid_idxs]

    # standardize
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
    else:
        scaler = None

    return X_train, y_train, X_valid, y_valid, scaler


def prepare_mil_data(train_file_idxs, valid_file_idxs, embeddings, target_list, standardize=True,
                     oversample=None, oversample_iters=1, thresh_type="mean"):
    """
    Prepare inputs and targets for MIL training using training and validation indices.

    Parameters
    ----------
    train_file_idxs
    valid_file_idxs
    embeddings
    target_list
    standardize
    oversample
    oversample_iters
    thresh_type

    Returns
    -------
    X_train
    y_train
    X_valid
    y_valid
    scaler

    """

    X_train_mil = np.array([embeddings[idx] for idx in train_file_idxs])
    X_valid_mil = np.array([embeddings[idx] for idx in valid_file_idxs])
    y_train_mil = np.array([target_list[idx] for idx in train_file_idxs])
    y_valid_mil = np.array([target_list[idx] for idx in valid_file_idxs])

    if oversample == 'mlsmote':
        X_train_mil, y_train_mil, _ = mlsmote(X_train_mil, y_train_mil, oversample_iters=oversample_iters,
                                              thresh_type=thresh_type)
    elif oversample == 'lssmote':
        X_train_mil, y_train_mil, _ = lssmote(X_train_mil, y_train_mil, oversample_iters=oversample_iters,
                                              thresh_type=thresh_type)
    elif oversample is not None:
        raise ValueError("Unknown oversample method: {}".format(oversample))

    # standardize
    if standardize:
        scaler = StandardScaler()
        scaler.fit(np.array([emb for emb_grp in X_train_mil for emb in emb_grp]))

        X_train_mil = [scaler.transform(emb_grp) for emb_grp in X_train_mil]
        X_valid_mil = [scaler.transform(emb_grp) for emb_grp in X_valid_mil]
    else:
        scaler = None

    train_mil_idxs = np.random.permutation(len(X_train_mil))

    X_train_mil = np.array(X_train_mil)[train_mil_idxs]
    y_train_mil = np.array(y_train_mil)[train_mil_idxs]
    X_valid_mil = np.array(X_valid_mil)

    return X_train_mil, y_train_mil, X_valid_mil, y_valid_mil, scaler


## GENERIC MODEL TRAINING

def train_mlp(model, x_train, y_train, x_val, y_val, output_dir, batch_size=64,
              num_epochs=100, patience=20, learning_rate=1e-4):
    """
    Train a MLP model with the given data.

    Parameters
    ----------
    model
    x_train
    y_train
    x_val
    y_val
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
        validation_data=(x_val, y_val), callbacks=cb, verbose=2)

    return history


## MODEL TRAINING

def train_framewise(annotation_path, emb_dir, output_dir, exp_id, label_mode="low", batch_size=64, test_ratio=0.1,
                  num_epochs=100, patience=20, learning_rate=1e-4, hidden_layer_size=128,
                  l2_reg=1e-5, standardize=True, oversample=None, oversample_iters=1, thresh_type="mean",
                  timestamp=None):
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
    oversample
    oversample_iters
    thresh_type
    timestamp

    Returns
    -------

    """
    file_list, high_target_list, low_target_list, train_file_idxs, eval_file_idxs = load_sonyc_data(annotation_path)

    if label_mode == "low":
        target_list = low_target_list
        labels = sonyc_data.LOW_LEVEL_LABELS
    elif label_mode == "high":
        target_list = high_target_list
        labels = sonyc_data.HIGH_LEVEL_LABELS
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)

    valid_file_idxs, test_file_idxs = get_val_test_file_split(file_list, eval_file_idxs, target_list,
         test_ratio=test_ratio)

    embeddings = load_embeddings(file_list, emb_dir)
    X_train, y_train, X_valid, y_valid, scaler = prepare_framewise_data(train_file_idxs, valid_file_idxs, embeddings,
                                                                        target_list, standardize=standardize,
                                                                        oversample=oversample,
                                                                        oversample_iters=oversample_iters,
                                                                        thresh_type=thresh_type)

    _, emb_size = X_train.shape

    model = construct_mlp_framewise(emb_size, num_classes, hidden_layer_size=hidden_layer_size, l2_reg=l2_reg)

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    results_dir = os.path.join(output_dir, exp_id, timestamp)

    history = train_mlp(model, X_train, y_train, X_valid, y_valid, results_dir, batch_size=batch_size,
              num_epochs=num_epochs, patience=patience, learning_rate=learning_rate)

    results = {}
    results['train'] = evaluate_framewise_model(embeddings, target_list, train_file_idxs, model, labels)
    results['valid'] = evaluate_framewise_model(embeddings, target_list, valid_file_idxs, model, labels)
    results['test'] = evaluate_framewise_model(embeddings, target_list, test_file_idxs, model, labels, scaler=scaler)
    results['history'] = history.history

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def train_mil(annotation_path, emb_dir, output_dir, exp_id, label_mode="low",
        batch_size=64, test_ratio=0.1, num_epochs=100,
        patience=20, learning_rate=1e-4, hidden_layer_size=128, l2_reg=1e-5, standardize=True,
        oversample=None, oversample_iters=1, thresh_type="mean", timestamp=None):
    """
    Train and evaluate a MIL MLP model.

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
    oversample
    oversample_iters
    thresh_type
    timestamp

    Returns
    -------

    """
    file_list, high_target_list, low_target_list, train_file_idxs, eval_file_idxs = load_sonyc_data(annotation_path)

    if label_mode == "low":
        target_list = low_target_list
        labels = sonyc_data.LOW_LEVEL_LABELS
    elif label_mode == "high":
        target_list = high_target_list
        labels = sonyc_data.HIGH_LEVEL_LABELS
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)

    valid_file_idxs, test_file_idxs = get_val_test_file_split(file_list, eval_file_idxs, target_list, test_ratio=test_ratio)

    embeddings = load_embeddings(file_list, emb_dir)

    X_train, y_train, X_valid, y_valid, scaler = prepare_mil_data(train_file_idxs, valid_file_idxs, embeddings,
                                                                  target_list, standardize=standardize,
                                                                  oversample=oversample, oversample_iters=oversample_iters,
                                                                  thresh_type=thresh_type)

    _, num_frames, emb_size = X_train.shape

    model = construct_mlp_mil(num_frames, emb_size, num_classes, hidden_layer_size=hidden_layer_size, l2_reg=l2_reg)

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    results_dir = os.path.join(output_dir, exp_id, timestamp)

    history = train_mlp(model, X_train, y_train, X_valid, y_valid, results_dir, batch_size=batch_size,
                        num_epochs=num_epochs, patience=patience, learning_rate=learning_rate)

    results = {}
    results['train'] = evaluate_mil_model(embeddings, target_list, train_file_idxs, model, labels)
    results['valid'] = evaluate_mil_model(embeddings, target_list, valid_file_idxs, model, labels)
    results['test'] = evaluate_mil_model(embeddings, target_list, test_file_idxs, model, labels, scaler=scaler)
    results['history'] = history.history

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


## MODEL EVALUATION

def evaluate_framewise_model(embeddings, target_list, test_file_idxs, model, labels, scaler=None, average='micro'):
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
    y_true = []

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

        y_pred_max.append(pred_frame.max(axis=0))
        y_pred_mean.append(pred_frame.mean(axis=0))
        y_pred_softmax.append((softmax(pred_frame, axis=0) * pred_frame).sum(axis=0))

        y_true.append(y_)

    y_true = np.round(np.array(y_true))
    y_pred_max = np.round(np.array(y_pred_max))
    y_pred_mean = np.round(np.array(y_pred_mean))
    y_pred_softmax = np.round(np.array(y_pred_softmax))

    results = {
        pool_method: {
            'predictions': y_pred.astype(int).tolist(),
            'file_idxs': np.array(test_file_idxs, dtype=int).tolist(),
            'overall_metrics': {},
            'class_metrics': {k: {} for k in labels}
        }
        for pool_method, y_pred in (
            ('max', y_pred_max), ('mean', y_pred_mean), ('softmax', y_pred_softmax))}

    results["max"]["overall_metrics"]["accuracy"] = binary_accuracy_round_np(y_true, y_pred_max)
    results["max"]["overall_metrics"]["precision"] = precision_score(y_true, y_pred_max, average=average)
    results["max"]["overall_metrics"]["recall"] = recall_score(y_true, y_pred_max, average=average)
    try:
        results["max"]["overall_metrics"]["auroc"] = roc_auc_score(y_true, y_pred_max, average=average)
        results["max"]["overall_metrics"]["f1"] = f1_score(y_true, y_pred_max, average=average)
        results["max"]["overall_metrics"]["map"] = average_precision_score(y_true, y_pred_max, average=average)
    except:
        results["max"]["overall_metrics"]["auroc"] = -1
        results["max"]["overall_metrics"]["f1"] = -1
        results["max"]["overall_metrics"]["map"] = -1

    results["mean"]["overall_metrics"]["accuracy"] = binary_accuracy_round_np(y_true, y_pred_mean)
    results["mean"]["overall_metrics"]["precision"] = precision_score(y_true, y_pred_mean, average=average)
    results["mean"]["overall_metrics"]["recall"] = recall_score(y_true, y_pred_mean, average=average)
    try:
        results["mean"]["overall_metrics"]["auroc"] = roc_auc_score(y_true, y_pred_mean, average=average)
        results["mean"]["overall_metrics"]["f1"] = f1_score(y_true, y_pred_mean, average=average)
        results["mean"]["overall_metrics"]["map"] = average_precision_score(y_true, y_pred_mean, average=average)
    except:
        results["mean"]["overall_metrics"]["auroc"] = -1
        results["mean"]["overall_metrics"]["f1"] = -1
        results["mean"]["overall_metrics"]["map"] = -1

    results["softmax"]["overall_metrics"]["accuracy"] = binary_accuracy_round_np(y_true, y_pred_softmax)
    results["softmax"]["overall_metrics"]["precision"] = precision_score(y_true, y_pred_softmax, average=average)
    results["softmax"]["overall_metrics"]["recall"] = recall_score(y_true, y_pred_softmax, average=average)
    try:
        results["softmax"]["overall_metrics"]["auroc"] = roc_auc_score(y_true, y_pred_softmax, average=average)
        results["softmax"]["overall_metrics"]["f1"] = f1_score(y_true, y_pred_softmax, average=average)
        results["softmax"]["overall_metrics"]["map"] = average_precision_score(y_true, y_pred_softmax, average=average)
    except:
        results["softmax"]["overall_metrics"]["auroc"] = -1
        results["softmax"]["overall_metrics"]["f1"] = -1
        results["softmax"]["overall_metrics"]["map"] = -1

    for idx, label in enumerate(labels):
        y_true_cls = y_true[:,idx]
        y_pred_max_cls = y_pred_max[:,idx]
        y_pred_mean_cls = y_pred_mean[:,idx]
        y_pred_softmax_cls = y_pred_softmax[:,idx]

        results["max"]["class_metrics"][label]["num_examples"] = y_true_cls.shape[0]
        results["max"]["class_metrics"][label]["num_positives"] = int((y_true_cls.astype(int) == 1).sum())
        results["max"]["class_metrics"][label]["accuracy"] = accuracy_score(y_true_cls, y_pred_max_cls)
        results["max"]["class_metrics"][label]["precision"] = precision_score(y_true_cls, y_pred_max_cls)
        results["max"]["class_metrics"][label]["recall"] = recall_score(y_true_cls, y_pred_max_cls)

        results["mean"]["class_metrics"][label]["num_examples"] = y_true_cls.shape[0]
        results["mean"]["class_metrics"][label]["num_positives"] = int((y_true_cls.astype(int) == 1).sum())
        results["mean"]["class_metrics"][label]["accuracy"] = accuracy_score(y_true_cls, y_pred_mean_cls)
        results["mean"]["class_metrics"][label]["precision"] = precision_score(y_true_cls, y_pred_mean_cls)
        results["mean"]["class_metrics"][label]["recall"] = recall_score(y_true_cls, y_pred_mean_cls)

        results["softmax"]["class_metrics"][label]["num_examples"] = y_true_cls.shape[0]
        results["softmax"]["class_metrics"][label]["num_positives"] = int((y_true_cls.astype(int) == 1).sum())
        results["softmax"]["class_metrics"][label]["accuracy"] = accuracy_score(y_true_cls, y_pred_softmax_cls)
        results["softmax"]["class_metrics"][label]["precision"] = precision_score(y_true_cls, y_pred_softmax_cls)
        results["softmax"]["class_metrics"][label]["recall"] = recall_score(y_true_cls, y_pred_softmax_cls)

        if np.all(y_true_cls) or np.all(np.round(1 - y_true_cls)):
            # If all one class, just set to -1
            results["max"]["class_metrics"][label]["f1"] = -1
            results["max"]["class_metrics"][label]["auroc"] = -1
            results["max"]["class_metrics"][label]["map"] = -1

            results["mean"]["class_metrics"][label]["f1"] = -1
            results["mean"]["class_metrics"][label]["auroc"] = -1
            results["mean"]["class_metrics"][label]["map"] = -1

            results["softmax"]["class_metrics"][label]["f1"] = -1
            results["softmax"]["class_metrics"][label]["auroc"] = -1
            results["softmax"]["class_metrics"][label]["map"] = -1
        else:
            results["max"]["class_metrics"][label]["f1"] = f1_score(y_true_cls, y_pred_max_cls)
            results["max"]["class_metrics"][label]["auroc"] = roc_auc_score(y_true_cls, y_pred_max_cls)
            results["max"]["class_metrics"][label]["map"] = average_precision_score(y_true_cls, y_pred_max_cls)

            results["mean"]["class_metrics"][label]["f1"] = f1_score(y_true_cls, y_pred_mean_cls)
            results["mean"]["class_metrics"][label]["auroc"] = roc_auc_score(y_true_cls, y_pred_mean_cls)
            results["mean"]["class_metrics"][label]["map"] = average_precision_score(y_true_cls, y_pred_mean_cls)

            results["softmax"]["class_metrics"][label]["f1"] = f1_score(y_true_cls, y_pred_softmax_cls)
            results["softmax"]["class_metrics"][label]["auroc"] = roc_auc_score(y_true_cls, y_pred_softmax_cls)
            results["softmax"]["class_metrics"][label]["map"] = average_precision_score(y_true_cls, y_pred_softmax_cls)

    return results


def evaluate_mil_model(embeddings, target_list, test_file_idxs, model, labels, scaler=None, average='micro'):
    """
    Evaluate the output of a MIL classification model.

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
    # Evaluate
    if scaler is None:
        X_test_mil = np.array([embeddings[idx] for idx in test_file_idxs])
    else:
        X_test_mil = np.array([scaler.transform(embeddings[idx]) for idx in test_file_idxs])

    y_test_mil = np.round(np.array([target_list[idx] for idx in test_file_idxs]))

    y_pred_mil = np.round(model.predict(X_test_mil))
    results = {
        "predictions": y_pred_mil.astype(int).tolist(),
        'file_idxs': np.array(test_file_idxs, dtype=int).tolist(),
        "overall_metrics": {},
        "class_metrics": {k: {} for k in labels},
        "alpha": {}
    }

    results["overall_metrics"]["accuracy"] = float(binary_accuracy_round_np(y_test_mil, y_pred_mil))
    results["overall_metrics"]["precision"] = float(precision_score(y_test_mil, y_pred_mil, average=average))
    results["overall_metrics"]["recall"] = float(recall_score(y_test_mil, y_pred_mil, average=average))
    try:
        results["overall_metrics"]["auroc"] = float(roc_auc_score(y_test_mil, y_pred_mil, average=average))
        results["overall_metrics"]["f1"] = float(f1_score(y_test_mil, y_pred_mil, average=average))
        results["overall_metrics"]["map"] = float(average_precision_score(y_test_mil, y_pred_mil, average=average))
    except:
        results["overall_metrics"]["auroc"] = -1
        results["overall_metrics"]["f1"] = -1
        results["overall_metrics"]["map"] = -1

    # Accuracy
    for idx, label in enumerate(labels):
        y_true_cls = y_test_mil[:,idx]
        y_pred_mil_cls = y_pred_mil[:,idx]
        results["class_metrics"][label]["num_examples"] = y_true_cls.shape[0]
        results["class_metrics"][label]["num_positives"] = int((y_true_cls.astype(int) == 1).sum())
        results["class_metrics"][label]["accuracy"] = accuracy_score(y_true_cls, y_pred_mil_cls)
        results["class_metrics"][label]["precision"] = precision_score(y_true_cls, y_pred_mil_cls)
        results["class_metrics"][label]["recall"] = recall_score(y_true_cls, y_pred_mil_cls)

        if np.all(y_true_cls) or np.all(np.round(1 - y_true_cls)):
            # If all one class, just set to -1
            results["class_metrics"][label]["f1"] = -1
            results["class_metrics"][label]["auroc"] = -1
            results["class_metrics"][label]["map"] = -1
        else:
            results["class_metrics"][label]["f1"] = f1_score(y_true_cls, y_pred_mil_cls)
            results["class_metrics"][label]["auroc"] = roc_auc_score(y_true_cls, y_pred_mil_cls)
            results["class_metrics"][label]["map"] = average_precision_score(y_true_cls, y_pred_mil_cls)

    # Alpha values
    for label, alpha in zip(labels, model.get_layer('auto_pool1d_1').get_weights()[0][0]):
        results["alpha"][label] = float(alpha)

    return results


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
    parser.add_argument("--test_ratio", type=float, default=0.5)
    parser.add_argument("--no_standardize", action='store_true')
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--label_mode", type=str, choices=["low", "high"],
                        default='low')
    parser.add_argument("--oversample", type=str, choices=["mlsmote", "lssmote"])
    parser.add_argument("--oversample_iters", type=int, default=1)
    parser.add_argument("--thresh_type", type=str, default="mean",
                        choices=["mean"] + ["percentile_{}".format(i) for i in range(1,100)])
    parser.add_argument("--target_mode", type=str, choices=["framewise", "mil"],
                        default='framewise')

    args = parser.parse_args()

    # save args to disk
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(args.output_dir, args.exp_id, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    kwarg_file = os.path.join(out_dir, "hyper_params.json")
    with open(kwarg_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.target_mode == 'mil':
        train_mil(args.annotation_path,
                  args.emb_dir,
                  args.output_dir,
                  args.exp_id,
                  label_mode=args.label_mode,
                  batch_size=args.batch_size,
                  test_ratio=args.test_ratio,
                  num_epochs=args.num_epochs,
                  patience=args.patience,
                  learning_rate=args.learning_rate,
                  hidden_layer_size=args.hidden_layer_size,
                  l2_reg=args.l2_reg,
                  standardize=(not args.no_standardize),
                  oversample=args.oversample,
                  oversample_iters=args.oversample_iters,
                  timestamp=timestamp)
    elif args.target_mode == 'framewise':
        train_framewise(args.annotation_path,
                  args.emb_dir,
                  args.output_dir,
                  args.exp_id,
                  label_mode=args.label_mode,
                  batch_size=args.batch_size,
                  test_ratio=args.test_ratio,
                  num_epochs=args.num_epochs,
                  patience=args.patience,
                  learning_rate=args.learning_rate,
                  hidden_layer_size=args.hidden_layer_size,
                  l2_reg=args.l2_reg,
                  standardize=(not args.no_standardize),
                    oversample=args.oversample,
                    oversample_iters=args.oversample_iters,
                  timestamp=timestamp)
