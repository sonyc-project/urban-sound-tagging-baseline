# Copy from https://github.com/sonyc-project/sensor-embedding/blob/master/sensor_embedding/downstream/sonyc_data.py
import csv
import re
import numpy as np
from collections import OrderedDict


FINE_PATTERN = r'((^\d+-[\dX]+)_.*)_presence$'
COARSE_PATTERN = r'((^\d+)_.*)_presence$'


def load_ust_data(filepath):
    """
    Load urban sound tagging annotation data from an annotation file.

    Parameters
    ----------
    filepath

    Returns
    -------
    annotation_data

    """
    data = OrderedDict()
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            audio_filename = row.pop('audio_filename')
            if audio_filename not in data:
                data[audio_filename] = []
            ann = {}
            for k, v in row.items():
                if not v.strip():
                    # TODO: HANDLE THIS
                    v = 0.0
                elif '_presence' in k:
                    v = float(v)
                elif v == '-1':
                    v = None
                ann[k] = v

            data[audio_filename].append(ann)

    return data


def _fine_label_order(x):
    subcode = x.split('_')[0].split('-')[-1]
    if subcode == 'X':
        return float('inf')
    else:
        return int(subcode)


def get_taxonomy(annotation_data):
    """
    Parse taxonomy from annotation data

    Parameters
    ----------
    annotation_data

    Returns
    -------
    taxonomy

    """
    fine_labels = OrderedDict()
    coarse_labels = OrderedDict()
    fields = next(iter(annotation_data.values()))[0].keys()

    for key in fields:
        fine_match = re.search(FINE_PATTERN, key)
        coarse_match = re.search(COARSE_PATTERN, key)

        if fine_match:
            fine_label, fine_code = fine_match.groups()
            fine_labels[fine_code] = fine_label
        elif coarse_match:
            coarse_label, coarse_code = coarse_match.groups()
            coarse_labels[coarse_code] = coarse_label

    taxonomy = OrderedDict((k, []) for k in coarse_labels.values())

    for fine_code, fine_label in fine_labels.items():
        coarse_code, sub_code = fine_code.split('-')
        taxonomy[coarse_labels[coarse_code]].append(fine_label)

    for coarse_label in taxonomy.keys():
        taxonomy[coarse_label] = sorted(taxonomy[coarse_label], key=_fine_label_order)

    return taxonomy


def get_subset_split(annotation_data):
    """
    Get indices for train and validation subsets

    Parameters
    ----------
    annotation_data

    Returns
    -------
    train_idxs
    valid_idxs

    """

    train_idxs = []
    valid_idxs = []
    for idx, file_anns in enumerate(annotation_data.values()):
        if file_anns[0]['split'] == 'train':
            train_idxs.append(idx)
        else:
            valid_idxs.append(idx)

    return np.array(train_idxs), np.array(valid_idxs)


def get_target(file_anns, labels):
    """
    Get file target annotation vector for the given set of labels

    Parameters
    ----------
    file_anns
    labels

    Returns
    -------
    target

    """
    target = []
    for label in labels:
        count = sum([ann[label + '_presence'] for ann in file_anns])
        if count > 0:
            target.append(1.0)
        else:
            target.append(0.0)

    return target


def get_file_targets(annotation_data, labels):
    """
    Get fine level targets for all files in the dataset

    Parameters
    ----------
    annotation_data
    labels

    Returns
    -------
    fine_targets

    """
    targets = []
    for file_anns in annotation_data.values():
        t = get_target(file_anns, labels)
        targets.append(t)

    return np.array(targets)
