import argparse
import json, hjson
import os
import re
import glob
import datetime
import sys
import random

import cv2 as cv
import numpy as np
import tqdm
import mmcv
from mmdet.utils import select_images
import torch

from utils import extract_lengths, writejson, compute_smallest_distances



def main(args):
    '''
    Extract frames with disagreement between AI and videocoding, for 14 reference videos.
    '''

    device = f'cuda:{args.gpu_id}'
    
    # load dicts to link predefined videos and their videocoding files
    videocoding_config =  hjson.load(open(args.videocoding_config, 'r'))
    json_dict = videocoding_config['json_dict']
    inputpath = videocoding_config['inputpath']

    ########### loop over selected videos : extract frames every nth meter from video, then find frames with disagreement between AI and videocoding ##############
    full_list = []
    FP_dict = {}
    FN_dict = {}

    # load dicts to link classes from AI and videocoding
    cls_config =  hjson.load(open(args.cls_config, 'r'))
    classes_vid = cls_config['classes_vid']
    classes_AI = cls_config['classes_AI']
    classes_comp= cls_config['classes_comp']
    cls_weight = cls_config['cls_weight']
    
    for vid, jjson in json_dict.items():
        print(f'\nProcessing {vid}')
        print(jjson)
        disagreement_dict = {'FP':{}, 'FN':{}}
        
        length_AI, length_AI_score, length_video, extract_path = extract_lengths(
            f'{inputpath}/{jjson}', f'{inputpath}/{vid}', f'{inputpath}/{vid.replace("mp4", "csv")}',
            classes_vid, classes_AI, classes_comp,
            'det', args.config, args.checkpoint, args.process_every_nth_meter,
            filter_road=False, device=device
        )
        
        # loop over classes
        for ic, (lai, score, lv) in enumerate(zip(length_AI, length_AI_score, length_video)):
            class_name = list(classes_comp.keys())[ic]
            
            lai = np.array(lai)
            score = np.array(score)
            lai_thr = lai[score > args.threshold_score]  # apply threshold to length_AI

            distances_AI = compute_smallest_distances(lai_thr, lv)
            distances_video = compute_smallest_distances(lv, lai_thr)

            # extract frames with disagreement between AI and videocoding
            # AI predictions but no videocoding annotations (FP)
            disagreement_dict['FP'][class_name] = np.sort(lai_thr[distances_AI>args.threshold_dist]).tolist()
            # videocoding annotations but no AI predictions (FN)
            disagreement_dict['FN'][class_name] = np.sort(np.array(lv)[distances_video>args.threshold_dist]).tolist()

        with open(f'{extract_path}/disagreement_dict.json', 'w') as fout:
            json.dump(disagreement_dict, fout, indent = 6)

        
        # Make list of frames with disagreement between AI and videocoding
        # False Positive: simply send images for annotation
        FP_list = []
        for cls, val in disagreement_dict['FP'].items():
            FP_list += val
            frames_list = [f'{extract_path}/{v}.jpg' for v in val]
            try:
                FP_dict[cls] += frames_list
            except:
                FP_dict[cls] = frames_list

        FP_list = [f'{extract_path}/{im}.jpg' for im in FP_list]
        full_list += FP_list

        # False Negatives: take saved frame closest to extracted length + surrounding frames
        saved_frames = glob.glob(f'{extract_path}/*jpg')
        saved_frames = np.sort(np.array([float(sf.replace(extract_path, '').replace('/', '').replace('.jpg', '')) for sf in saved_frames]))

        FN_list = []
        for cls, val in disagreement_dict['FN'].items():
            frames_list = []
            for v in val:
                idx = np.argmin(np.abs(saved_frames - v))
                FN_list.append(f'{extract_path}/{saved_frames[idx]}.jpg')
                frames_list.append(f'{extract_path}/{saved_frames[idx]}.jpg')                    
            # saved frames corresponding to false negatives
            try:
                FN_dict[cls] += frames_list
            except:
                FN_dict[cls] = frames_list

        full_list += FN_list
            
    full_list = list(set(full_list))
    print(len(full_list))

    ########### load AI model #####################
    import mmdet.apis
    model = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device=device,
    )

    
    # copy images with disagreement and pre-annotate them with detection model
    dt = datetime.datetime.now()
    timestamp = f'{dt.year}{dt.month}{dt.day}_{dt.hour}{dt.minute}'
    outpath = f'prepare_annotations/images_to_annotate_{timestamp}'
    os.makedirs(f'{outpath}/FN/', exist_ok=True)
    os.makedirs(f'{outpath}/FP/', exist_ok=True)

    for im in full_list:
        # find if image is FN
        cls_list = []
        for cls, images in FN_dict.items():
            if im in images:
                cls_list.append(cls)
        
        video_name = im.split('/')[-3]
        im_name = int(float(im.split('/')[-1].replace('.jpg', '')))
        im_path = im.replace(' ', '\ ')
        if len(cls_list) == 0:
            for cls, images in FP_dict.items():
                if im in images:
                    cls_list.append(cls)
            FP_name = '_'.join(cls_list)
            os.makedirs(f'{outpath}/FP/{FP_name}', exist_ok=True)
            im_name_new = f'{outpath}/FP/{FP_name}/{video_name.replace(" ", "")}_{im_name}.jpg'
        else:
            FN_name = '_'.join(cls_list)
            os.makedirs(f'{outpath}/FN/{FN_name}', exist_ok=True)
            im_name_new = f'{outpath}/FN/{FN_name}/{video_name.replace(" ", "")}_{im_name}.jpg'
        os.system(f'cp {im_path} {im_name_new}')

        image = cv.imread(im_name_new)

        ann = []
            
        try:
            res = mmdet.apis.inference_detector(model, image)
        except:
            print(im_name_new)
            os.system(f'rm "{im_name_new}"')
            continue
        image_width = image.shape[1]
        image_height = image.shape[0]
        for ic, c in enumerate(res): # loop on classes
            for ip, p in enumerate(c): # loop on bboxes
                x1, y1, x2, y2 = p[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if p[4] > 0.3:
                    ann.append(f'{ic} {(x1+x2)/2/image_width} {(y1+y2)/2/image_height} {(x2-x1)/image_width} {(y2-y1)/image_height}')
                        
        ann_name = im_name_new.replace('.jpg', '.txt')
        with open(ann_name, 'w') as f:
            for a in ann:
                f.write(f'{a}\n')
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath-mp4', help='path to input videos.')
    parser.add_argument('--inputpath-json', help='path to input videocoding annotations.')
    
    parser.add_argument('--config', required=True, help='config for detection model.')
    parser.add_argument('--checkpoint', required=True, help='weights for detection model.')
    
    parser.add_argument('--cls-config', default='configs/classes_reference_videos.json', help='json file with dicts to link classes from AI and videocoding.')
    parser.add_argument('--videocoding-config', default='configs/videocoding_reference_videos.json', help='json file with dicts to videocoding files corresponding to predefined videos.')
    
    parser.add_argument('--process-every-nth-meter', type=float, default=3, help='step in meters between processed frames.')
    parser.add_argument('--threshold-dist', type=float, default=10, help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')
    parser.add_argument('--threshold-score', type=float, default=0.3, help='score threshold to be used for disagreement extraction.')
    parser.add_argument('--n-sel-al', type=float, default=1000, help='Number of frames to select with Active Learning for annotation.')

    parser.add_argument('--gpu-id', type=int, default=0)
    
    args = parser.parse_args()

    main(args)
    
    
