import argparse
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Evaluation script for Urban Sound Tagging task for the DCASE 2019 Challenge.

        See `metrics.py` for more information about the metrics.
        """)

    parser.add_argument('prediction_path', type=str,
                        help='Path to prediction CSV file.')
    parser.add_argument('annotation_path', type=str,
                        help='Path to dataset annotation CSV file.')
    parser.add_argument('yaml_path', type=str,
                        help='Path to dataset taxonomy YAML file.')

    args = parser.parse_args()

    for mode in ("fine", "coarse"):

        df_dict = evaluate(args.prediction_path,
                           args.annotation_path,
                           args.yaml_path,
                           mode)

        micro_auprc = micro_averaged_auprc(df_dict)
        macro_auprc = macro_averaged_auprc(df_dict)

        print("{} level evaluation:".format(mode.capitalize()))
        print("======================")
        print(" * Micro AUPRC: {}".format(micro_auprc))
        print(" * Macro AUPRC: {}".format(macro_auprc))
