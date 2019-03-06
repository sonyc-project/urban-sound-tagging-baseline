import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import yaml


def evaluate_fine(
        Y_true, Y_pred, is_true_incomplete, is_pred_incomplete):
    """
    Counts overall numbers of true positives (TP), false positives (FP),
    and false negatives (FN) in the predictions of a system, for a number K
    of fine-level classes within a given coarse category, in a dataset of N
    different samples. In addition to the K so-called "complete" tags (i.e.
    with a determinate fine-level category as well as a determinate
    coarse-level category), we consider the potential presence of an "incomplete"
    tag, i.e. denoting the presence of a class with a determinate coarse-level
    category yet no determinate fine-level category. This incomplete tag
    be present in either the prediction or the ground truth.

    Our method for evaluating a multilabel classifier on potentially incomplete
    knowledge of the ground truth consists of two parts, which are ultimately
    aggregated into a single count.

    For the samples with complete knowledge of both ground truth (Part I in the
    code below), we simply apply classwise Boolean logic to compute TP, FP, and
    FN independently for every fine-level tag, and finally aggregate across
    all tags.

    However, for the samples with incomplete knowledge of the ground truth
    (Part II in the code below), we perform a "coarsening" of the prediction by
    apply a disjunction on the fine-level complete tags as well as the
    coarse incomplete tag. If that coarsened prediction is positive, the sample
    produces a true positive; otherwise, it produces a false negative.

    Samples which contain the incomplete tag in the prediction but not the
    ground truth overlap Parts I and II. In this case, we sum the zero, one,
    or multiple false alarm(s) from Part I with the one false alarm from Part II
    to produce a final number of false positives FP.

    Parameters
    ----------
    Y_true: array of bool, shape = [n_samples, n_classes]
        One-hot encoding of true presence for complete fine tags.
        Y_true[n, k] is equal to 1 if the class k is truly present in sample n,
        and equal to 0 otherwise.

    Y_pred: array of bool, shape = [n_samples, n_classes]
        One-hot encoding of predicted class presence for complete fine tags.
        Y_true[n, k] is equal to 1 if the class k is truly present in sample n,
        and equal to 0 otherwise.

    is_true_incomplete: array of bool, shape = [n_samples]
        One-hot encoding of true presence for the incomplete fine tag.
        is_true[n] is equal to 1 if an item that truly belongs to the
        coarse category at hand, but the fine-level tag of that item is
        truly uncertain, or truly unlike any of the K available fine tags.

    is_pred_incomplete: array of bool, shape = [n_samples]
        One-hot encoding of predicted presence for the incomplete fine tag.
        is_true[n] is equal to 1 if the system predicts the existence of an
        item that does belongs to the coarse category at hand, yet its
        fine-level tag of that item is uncertain or unlike any of the
        K available fine tags.

    Returns
    -------
    TP: int
        Number of true positives.

    FP: int
        Number of false positives.

    FN: int
        Number of false negatives.
    """

    ## PART I. SAMPLES WITH COMPLETE GROUND TRUTH AND COMPLETE PREDICTION
    # Negate the true_incomplete Boolean and replicate it K times, where
    # K is the number of fine tags.
    # For each sample and fine tag, this mask is equal to 0 if the
    # ground truth contains the incomplete fine tag and 1 if the ground
    # truth does not contain the incomplete fine tag.
    # The result is a (N, K) matrix.
    is_true_complete = np.tile(np.logical_not(
        is_true_incomplete)[:, np.newaxis], (1, Y_pred.shape[1]))

    # Compute true positives for samples with complete ground truth.
    # For each sample n and each complete tag k, is_TP_complete is equal to 1
    # if and only if the following three conditions are met:
    # (i)   the ground truth of sample n is complete
    # (ii)  the ground truth of sample n contains complete fine tag k
    # (iii) the prediction of sample n contains complete fine tag k
    # The result is a (N, K) matrix.
    is_TP_complete = np.logical_and.reduce((Y_true, Y_pred, is_true_complete))

    # Compute false positives for samples with complete ground truth.
    # For each sample n and each complete tag k, is_FP_complete is equal to 1
    # if and only if the following three conditions are met:
    # (i)   the ground truth of sample n is complete
    # (ii)  the ground truth of sample n does not contain complete fine tag k
    # (iii) the prediction of sample n contains complete fine tag k
    # The result is a (N, K) matrix.
    is_FP_complete = np.logical_and.reduce(
        (np.logical_not(Y_true), Y_pred, is_true_complete))

    # Compute false negatives for samples with complete ground truth.
    # For each sample n and each complete tag k, is_FN_complete is equal to 1
    # if and only if the following three conditions are met:
    # (i)   the ground truth of sample n is is complete
    # (ii)  the ground truth of sample n contains complete fine tag k
    # (iii) the prediction of sample n does not contain complete fine tag k
    # The result is a (N, K) matrix.
    is_FN_complete = np.logical_and.reduce(
        (Y_true, np.logical_not(Y_pred), is_true_complete))


    ## PART II. SAMPLES WITH INCOMPLETE GROUND TRUTH OR INCOMPLETE PREDICTION.
    # Compute a vector of "coarsened predictions".
    # For each sample, the coarsened prediction is equal to 1 if any of the
    # complete fine tags is predicted as present, or if the incomplete fine
    # tag is predicted as present. Conversely, it is set equal to 1 if all
    # of the complete fine tags are predicted as absent, and if the incomplete
    # fine tags are predicted as absent.
    # The result is a (N,) vector.
    y_pred_coarsened_without_incomplete = np.logical_or.reduce(Y_pred, axis=1)
    y_pred_coarsened = np.logical_or(
        (y_pred_reduced_without_incomplete, is_pred_incomplete))

    # Compute true positives for samples with incomplete ground truth.
    # For each sample n, is_TP_incomplete is equal to 1
    # if and only if the following two conditions are met:
    # (i)   the ground truth contains the incomplete fine tag
    # (ii)  the coarsened prediction of sample n contains at least one tag
    # The result is a (N,) vector.
    is_TP_incomplete = np.logical_and((is_true_incomplete, y_pred_reduced))

    # Compute false positives for samples with incomplete ground truth.
    # For each sample n, is_FP_incomplete is equal to 1
    # if and only if the following two conditions are met:
    # (i)   the incomplete fine tag is absent in the ground truth
    # (ii)  the prediction of sample n contains the incomplete fine tag
    # The result is a (N,) vector.
    is_FP_incomplete = np.logical_and(
        (np.logical_not(is_true_incomplete), y_pred_incomplete))

    # Compute false negatives for samples with incomplete ground truth.
    # For each sample n, is_FN_incomplete is equal to 1
    # if and only if the following two conditions are met:
    # (i)   the incomplete fine tag is present in the ground truth
    # (ii)  the coarsened prediction of sample n does not contain any tag
    # The result is a (N,) vector.
    is_FN_incomplete = np.logical_and(
        (is_true_incomplete, np.logical_not(y_pred_reduced)))


    ## PART IV. AGGREGATE EVALUATION OF ALL SAMPLES
    # The following three sums are performed over NxK Booleans,
    # implicitly converted as integers 0 (False) and 1 (True).
    TP_complete = sum(is_TP_complete)
    FP_complete = sum(is_FP_complete)
    FN_complete = sum(is_FN_complete)

    # The following three sums are performed over N Booleans,
    # implicitly converted as integers 0 (False) and 1 (True).
    TP_incomplete = sum(is_TP_incomplete)
    FP_incomplete = sum(is_FP_incomplete)
    FN_incomplete = sum(is_FN_incomplete)

    # Sum FP, TP, and FN for samples that have complete ground truth
    # with FP, TP, and FN for samples that have incomplete ground truth.
    TP = TP_complete + TP_incomplete
    FP = FP_complete + is_FP_incomplete
    FN = FN_complete + is_FN_incomplete
    return TP, FP, FN


def evaluate_coarse(y_true, y_pred):
    """
    Counts overall numbers of true positives (TP), false positives (FP),
    and false negatives (FN) in the predictions of a system, for a single
    Boolean attribute, in a dataset of N different samples.


    Parameters
    ----------
    y_true: array of bool, shape = [n_samples,]
        One-hot encoding of true presence for a given coarse tag.
        y_true[n] is equal to 1 if the tag is present in the sample.

    y_pred: array of bool, shape = [n_samples,]
        One-hot encoding of predicted presence for a given coarse tag.
        y_pred[n] is equal to 1 if the tag is present in the sample.


    Returns
    -------
    TP: int
        Number of true positives.

    FP: int
        Number of false positives.

    FN: int
        Number of false negatives.
    """
    cm = confusion_matrix(y_true, y_pred)
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    return TP, FP, FN


def parse_fine_prediction(pred_csv_path, yaml_path):
    """
    Parse fine-level predictions from a CSV file containing both fine-level
    and coarse-level predictions (and possibly additional metadata).
    Returns a Pandas DataFrame in which the column names are mixed (coarse-fine)
    IDs of the form 1-1, 1-2, 1-3, ..., 1-X, 2-1, 2-2, 2-3, ... 2-X, 3-1, etc.


    Parameters
    ----------
    pred_csv_path: string
        Path to the CSV file containing predictions.

    yaml_path: string
        Path to the YAML file containing fine taxonomy.


    Returns
    -------
    pred_fine_df: DataFrame
        Fine-level complete predictions.
    """

    # Create dictionary to parse tags
    with open(yaml_path, 'r') as stream:
        try:
            yaml_dict = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Collect tag names as strings and map them to mixed (coarse-fine) ID pairs.
    # The "mixed key" is a hyphenation of the coarse ID and fine ID.
    fine_dict = {}
    for coarse_id in yaml_dict["fine"]:
        for fine_id in yaml_dict["fine"][coarse_id]:
            mixed_key = "-".join([str(coarse_id), str(fine_id)])
            fine_dict[mixed_key] = yaml_dict["fine"][coarse_id][fine_id]

    # Invert the key-value relationship between mixed key and tag.
    # Now, tags are the keys, and mixed keys (coarse-fine IDs) are the values.
    # This is possible because tags are unique.
    rev_fine_dict = {fine_dict[k]: k for k in fine_dict}

    # Read comma-separated values with the Pandas library
    pred_df = pd.read_csv(pred_csv_path)

    # Assign a predicted column to each mixed key, by using the tag as an
    # intermediate hashing step.
    pred_fine_dict = {rev_fine_dict[f]: pred_df[f] for f in rev_fine_dict}

    # Loop over coarse tags.
    n_samples = len(pred_df)
    coarse_dict = yaml_dict["coarse"]
    for coarse_id in yaml_dict["coarse"]:
        # Construct incomplete fine tag by appending -X to the coarse tag.
        incomplete_tag = str(coarse_id) + "-X"

        # If the incomplete tag is not in the prediction, append a column of zeros.
        # This is the case e.g. for coarse ID 7 ("dogs") which has a single
        # fine-level tag ("7-1_dog-barking-whining") and thus no incomplete
        # tag 7-X.
        if incomplete_tag not in fine_dict.keys():
            pred_fine_dict[incomplete_tag] =\
                np.zeros((n_samples,)).astype('int')


    # Copy over the audio filename strings corresponding to each sample.
    pred_fine_dict["audio_filename"] = pred_df["audio_filename"]

    # Build a new Pandas DataFrame with mixed keys as column names.
    pred_fine_df = pd.DataFrame.from_dict(pred_fine_dict)

    # Return output in DataFrame format.
    # Column names are 1-1, 1-2, 1-3 ... 1-X, 2-1, 2-2, 2-3 ... 2-X, 3-1, etc.
    return pred_fine_df


def parse_coarse_prediction(pred_csv_path, yaml_path):
    """
    Parse coarse-level predictions from a CSV file containing both fine-level
    and coarse-level predictions (and possibly additional metadata).
    Returns a Pandas DataFrame in which the column names are coarse
    IDs of the form 1, 2, 3 etc.


    Parameters
    ----------
    pred_csv_path: string
        Path to the CSV file containing predictions.

    yaml_path: string
        Path to the YAML file containing coarse taxonomy.


    Returns
    -------
    pred_coarse_df: DataFrame
        Coarse-level complete predictions.
    """

    # Create dictionary to parse tags
    with open(yaml_path, 'r') as stream:
        try:
            yaml_dict = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Collect tag names as strings and map them to coarse ID pairs.
    rev_coarse_dict = {
        "".join(yaml_dict["coarse"][k].split("-")): k for k in yaml_dict["coarse"]}

    # Assign a predicted column to each coarse key, by using the tag as an
    # intermediate hashing step.
    pred_coarse_dict = {rev_coarse_dict[c]: pred_df[c] for c in rev_coarse_dict}

    # Copy over the audio filename strings corresponding to each sample.
    pred_coarse_dict["audio_filename"] = pred_df["audio_filename"]

    # Build a new Pandas DataFrame with coarse keys as column names.
    pred_coarse_df = pd.DataFrame.from_dict(pred_coarse_dict)

    # Return output in DataFrame format.
    # The column names are of the form 1, 2, 3, etc.
    pred_coarse_df = pred_coarse_df[coarse_columns]


def parse_ground_truth(annotation_path, yaml_path):
    """
    Parse ground truth annotations from a CSV file containing both fine-level
    and coarse-level predictions (and possibly additional metadata).
    Returns a Pandas DataFrame in which the column names are coarse
    IDs of the form 1, 2, 3 etc.


    Parameters
    ----------
    pred_csv_path: string
        Path to the CSV file containing predictions.

    yaml_path: string
        Path to the YAML file containing coarse taxonomy.


    Returns
    -------
    gt_df: DataFrame
        Ground truth.
    """
    # Create dictionary to parse tags
    with open(yaml_path, 'r') as stream:
        try:
            yaml_dict = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Load CSV file into a Pandas DataFrame.
    ann_df = pd.read_csv(annotation_path)

    # Restrict to ground truth ("annotator zero").
    ann_df = ann_df[ann_df["annotator_id"]=0]

    # Rename coarse columns.
    coarse_renaming = {
        "_".join(["high", "".join(coarse_dict[c].split("-")), "presence"]): str(c)
        for c in coarse_dict}
    gt_df = gt_df.rename(columns=coarse_renaming)

    # Rename fine columns.
    fine_renaming = {"_".join(["low", fine_dict[k], "presence"]): k
        for k in fine_dict}
    gt_df = gt_df.rename(columns=fine_renaming)

    # Loop over coarse tags.
    n_samples = len(gt_df)
    coarse_dict = yaml_dict["coarse"]
    for coarse_id in yaml_dict["coarse"]:
        # Construct incomplete fine tag by appending -X to the coarse tag.
        incomplete_tag = str(coarse_id) + "-X"

        # If the incomplete tag is not in the prediction, append a column of zeros.
        # This is the case e.g. for coarse ID 7 ("dogs") which has a single
        # fine-level tag ("7-1_dog-barking-whining") and thus no incomplete
        # tag 7-X.
        if incomplete_tag not in gt_df.columns:
            gt_df[incomplete_tag] = np.zeros((n_samples,)).astype('int')
