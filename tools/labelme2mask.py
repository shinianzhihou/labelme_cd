import argparse
import os
import json
import glob

import cv2
import tqdm
import numpy as np


def get_info_from_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    polygons = [np.array(item['points']).astype(np.int32) for item in shapes]
    height = data['imageHeight']
    width = data['imageWidth']
    return polygons, (height, width)



def run(args):
    data_dir = args.input_dir
    save_dir = args.output_dir

    files = glob.glob(os.path.join(data_dir,f'*.json'))
    for json_path in tqdm.tqdm(files,total=len(files)):

        polygons, (h,w) = get_info_from_json(json_path)
        mask = np.zeros((h,w))
        cv2.fillPoly(mask, polygons, 255)
        if save_dir:
            bname = os.path.basename(json_path).replace('.json','.png')
            mask_path = os.path.join(save_dir, bname)
        else:
            mask_path = json_path.replace('.json','.png')
        cv2.imwrite(mask_path,mask)
    return



def main():
    parser = argparse.ArgumentParser(
        description='convert labelme(json) to mask.')
    parser.add_argument('-i','--input_dir',type=str,help='images in input dir with specific suffix will be converted.')
    parser.add_argument('-o','--output_dir',default='',type=str,help='where to save the outputs.')
    parser.add_argument('-f','--factor',default=1, type=int,help='value for changed regions.')
    
    args = parser.parse_args()
    if args.output_dir and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    run(args)


if __name__ == '__main__':
    main()

