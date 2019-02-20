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


LOW_LEVEL_LABELS = [y for x in taxonomy.values() for y in x]
HIGH_LEVEL_LABELS = list(taxonomy.keys())


def load_sonyc_data(annotation_path):
    """
    Load SONYC annotation data from an annotation file.

    Parameters
    ----------
    annotation_path

    Returns
    -------
    file_list
    high_target_list
    low_target_list
    train_idxs
    eval_idxs

    """
    file_set = set()
    file_list = []
    train_idxs = []
    eval_idxs = []
    high_level_anns = {}
    low_level_anns = {}

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

                low_level_anns[filename] = []
                high_level_anns[filename] = []
                idx += 1

            low_distr = []
            for low_label in LOW_LEVEL_LABELS:
                val = float(row['low_' + low_label + '_presence'])
                low_distr.append(val)
            low_distr = np.clip(np.array(low_distr), 0, 1)
            low_level_anns[filename].append(low_distr)


            high_distr = []
            for high_label in HIGH_LEVEL_LABELS:
                val = float(row['high_' + high_label + '_presence'])
                high_distr.append(val)
            high_distr = np.clip(np.array(high_distr), 0, 1)
            high_level_anns[filename].append(high_distr)

    high_target_list = []
    low_target_list = []

    for filename in file_list:
        low_target = np.array(low_level_anns[filename]).mean(axis=0)
        low_target_list.append(low_target)
        high_target = np.array(high_level_anns[filename]).mean(axis=0)
        high_target_list.append(high_target)

    return file_list, high_target_list, low_target_list, train_idxs, eval_idxs
