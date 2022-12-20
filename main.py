# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import os
from tqdm import tqdm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = "C:/Users/zhujo/OneDrive/Documents/squares"

    avg_size = None
    all_squares = True
    one_color = True
    min_chip_height = 10000
    max_chip_height = 0
    min_img_height = 10000
    max_img_height = 0
    # cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    for dtype in ["train", "val"]:
        for direc in ["a", "b", "c"]:
            for file in tqdm(os.listdir(os.path.join(root, dtype, direc)), "Iterating"):
                if file != ".DS_Store":
                    filename = os.path.join(root, dtype, direc, file)
                    img = cv2.imread(filename, -1)
                    center = img.shape[0] // 2
                    main_color = img[center, center]
                    mask = np.all(img != 255, axis=-1)
                    idxs = np.where(mask)
                    y1, y2 = np.min(idxs[0]), np.max(idxs[0])
                    x1, x2 = np.min(idxs[1]), np.max(idxs[1])
                    width = x2 - x1
                    height = y2 - y1
                    # Check size of mask
                    img_size = img.shape[0]
                    chip_size = (width+height) // 2

                    min_chip_height = min(chip_size, min_chip_height)
                    max_chip_height = max(chip_size, max_chip_height)
                    min_img_height = min(img_size, min_img_height)
                    max_img_height = max(img_size, max_img_height)

                    # cv2.imshow("debug", img)
                    # cv2.waitKey(0)
    print(min_chip_height)
    print(max_chip_height)
    print(min_img_height)
    print(max_img_height)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
