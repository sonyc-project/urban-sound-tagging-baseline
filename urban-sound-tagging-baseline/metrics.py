def evaluate_fine(y_true, y_pred, is_true_incomplete, is_pred_incomplete):
    mask = np.tile(np.logical_not(
        is_true_incomplete)[:, np.newaxis], (1, y_pred.shape[1]))
    FP = np.sum(np.logical_and.reduce(
        (np.logical_not(y_true), mask, y_pred)))
    FN = np.sum(np.logical_and.reduce(
        (y_true, gt_notX, np.logical_not(y_pred)))) +\
              np.sum(np.logical_and(
        is_true_incomplete, np.logical_not(reduced_y_pred)))
    TP = np.sum(np.logical_and.reduce((y_true, mask, y_pred))) +\
               np.sum(np.logical_and(is_true_incomplete, reduced_y_pred))
    return TP, FP, FN


def evaluate_coarse(y_true, y_pred):
    coarse_confusion_matrix =\
        confusion_matrix(y_true, bool_pred)
    FP = coarse_confusion_matrix[0, 1])
    FN = coarse_confusion_matrix[1, 0])
    TP = coarse_confusion_matrix[1, 1])
    return TP, FP, FN
