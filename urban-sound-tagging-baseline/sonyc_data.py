# Copy from https://github.com/sonyc-project/sensor-embedding/blob/master/sensor_embedding/downstream/sonyc_data.py
import numpy as np
import csv
from collections import OrderedDict

taxonomy = OrderedDict([
    ('engine', ['smallsoundingengine',
               'mediumsoundingengine',
               'largesoundingengine',
               'engineofuncertainsize']),

    ('impactsound', ['rockdrill',
                    'jackhammer',
                    'hoeram',
                    'piledriver',
                    'otherunknownimpactmachinery',
                    'nonmachineryimpactsound']),

    ('poweredsaw', ['chainsaw',
                   'smallmediumrotatingsaw',
                   'largerotatingsaw',
                   'otherunknownpoweredsaw']),

    ('alertsignal', ['carhorn',
                    'caralarm',
                    'siren',
                    'reversebeeper',
                    'otherunknownalertsignal']),

    ('music', ['stationarymusic',
              'mobilemusic',
              'icecreamtruck',
              'musicfromuncertainsource']),

    ('humanvoice', ['personorsmallgrouptalking',
                   'personorsmallgroupshouting',
                   'largecrowd',
                   'amplifiedspeech',
                   'otherunknownhumanvoice']),

    ('dogbarkingwhining', ['dogbarkingwhining']),

    ('sensorfault', ['sensorfault']),

    ('otherunknownconstructionsound', ['otherunknownconstructionsound'])
])


ALL_FINE_LEVEL_LABELS = [y for x in taxonomy.values() for y in x]
ALL_COARSE_LEVEL_LABELS = list(taxonomy.keys())

FINE_LEVEL_LABELS = [x for x in ALL_FINE_LEVEL_LABELS
                    if "uncertain" not in x and "other" not in x]
COARSE_LEVEL_LABELS = [x for x in ALL_COARSE_LEVEL_LABELS
                    if "uncertain" not in x and "other" not in x]


def load_sonyc_data(annotation_path):
    """
    Load SONYC annotation data from an annotation file.

    Parameters
    ----------
    annotation_path

    Returns
    -------
    file_list
    coarse_target_list
    fine_target_list
    train_idxs
    eval_idxs

    """
    file_set = set()
    file_list = []
    train_idxs = []
    eval_idxs = []
    coarse_level_anns = {}
    fine_level_anns = {}

    with open(annotation_path, 'r') as f:
        reader = csv.DictReader(f)

        idx = 0
        for row in reader:
            filename = row['audio_filename']
            if filename not in file_set:
                file_set.add(filename)
                file_list.append(filename)
                if row['split'] == 'train':
                    train_idxs.append(idx)
                else:
                    eval_idxs.append(idx)

                fine_level_anns[filename] = []
                coarse_level_anns[filename] = []
                idx += 1

            fine_distr = []
            for fine_label in FINE_LEVEL_LABELS:
                val = float(row['low_' + fine_label + '_presence'])
                fine_distr.append(val)
            fine_distr = np.clip(np.array(fine_distr), 0, 1)
            fine_level_anns[filename].append(fine_distr)


            coarse_distr = []
            for coarse_label in COARSE_LEVEL_LABELS:
                val = float(row['high_' + coarse_label + '_presence'])
                coarse_distr.append(val)
            coarse_distr = np.clip(np.array(coarse_distr), 0, 1)
            coarse_level_anns[filename].append(coarse_distr)

    coarse_target_list = []
    fine_target_list = []

    for filename in file_list:
        fine_target = np.array(fine_level_anns[filename]).mean(axis=0)
        fine_target_list.append(fine_target)
        coarse_target = np.array(coarse_level_anns[filename]).mean(axis=0)
        coarse_target_list.append(coarse_target)

    return file_list, coarse_target_list, fine_target_list, train_idxs, eval_idxs
