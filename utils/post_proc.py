import cv2
import numpy as np
import scipy.stats


def uniform_label(label, img_line, thresh=144):
    """uniform label in each closed region
    
    Arguments:
        label {np.ndarray} -- shape [H, W]
        img_line {np.ndarray} -- shape [H, W]
    
    Keyword Arguments:
        thresh {int} -- threshold value for binarization (default: {144})
    
    Returns:
        label -- shape [H, W]
    """
    ret, img_bin = cv2.threshold(img_line, thresh, 255, 0)
    label = label.copy()
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        mask = np.zeros(img_line.shape, dtype=np.uint8)

        cv2.drawContours(mask ,contours, i, (255,), -1)
        j = hierarchy[0, i, 2]
        while(j != -1):
            cv2.drawContours(mask, contours, j,(0,), -1)
            j = hierarchy[0, j, 0] # next

        mask = mask.astype(np.bool)
        result = scipy.stats.mode(label[mask])
        label[mask] = result.mode
    
    return label
