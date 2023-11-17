import os
import json
import pandas as pd
from ast import literal_eval
import numpy as np
import glob
import shutil

from cocoviewer import Data, open_image, draw_masks, parse_coco


if __name__ == '__main__':
    # 示例用法
    data_dir = 'F:/PapersWithCode/robotData'
    img_dir = os.path.join(data_dir, 'images')
    annos_file = os.path.join(data_dir, 'annotations-1.json')
    mask_dir = os.path.join(data_dir, 'masks')
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    else:
        shutil.rmtree(mask_dir)
        os.mkdir(mask_dir)

    cat_id_color_file = './cat_id_color.txt'
    
    cocoData = Data(img_dir, annos_file)
    instances, id_files, categories = parse_coco(annos_file)

    df_cat = pd.read_csv(cat_id_color_file, sep=' ', header=None, names=['id', 'cat', 'color'])
    df_cat['color'] =  df_cat['color'].apply(literal_eval) # 将字符串列表转为实际列表
    for k in categories.keys():
        categories[k][1] = df_cat['color'].iloc[k]
        
    for id, f in id_files:
        img_open, draw_layer, draw = open_image(os.path.join(data_dir, f))
        img_name = os.path.basename(f)

        objects = [obj for obj in instances["annotations"] if obj["image_id"] == id]
    
        names_colors = categories.values()
        draw_masks(draw, objects, names_colors, ignore=[], alpha=128)

        draw_layer.save(os.path.join(mask_dir, 'mask_'+img_name))