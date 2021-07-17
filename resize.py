# coding=gbk
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse




data_path = '/Users/zhangyu/python/cyclegan/datasets/bitmoji/testOne'

for idx, img_name in enumerate(tqdm(os.listdir(data_path))):
    img = cv2.imread(os.path.join(data_path, img_name))
    if img is None:
       continue
    else:
        size = (256, 256)
        newImg = cv2.resize(img, size)
        # cv2.imwrite(os.path.join((args.save_path), img_name), newImg)
        cv2.imwrite(data_path+'/new_'+ img_name, newImg)
        print('write ',img_name)


