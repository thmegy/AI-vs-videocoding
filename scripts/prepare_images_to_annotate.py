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


def revert_dict(FP_dict, FN_dict, im_list):
    rev_dict = {}
    for im in im_list:
        cls_list = []
        im_name = float(im.split('/')[-1].replace('.jpg', ''))
        for cls, images in FP_dict.items():
            if im_name in images:
                cls_list.append(cls)
                
        for cls, images in FN_dict.items():
            if im in images:
                cls_list.append(cls)
        rev_dict[im] = cls_list
        
    return rev_dict



def main(args):
    '''
    Select 8 videocoded videos from input directory, find frames with disagreement between AI and videocoding, apply active learning to keep only N frames to annotate
    '''

    device = f'cuda:{args.gpu_id}'
    
    # make list of videos available
    input_vid_list = glob.glob(f'{args.inputpath_mp4}/*mp4')
    input_vid_list_tmp = input_vid_list.copy()
    
    # removed videos already used to extract images for annotation
    already_processed_videos = hjson.load(open('prepare_annotations/already_processed_videos.json', 'r'))
    for vid in input_vid_list:
        if vid.split('/')[-1] in already_processed_videos:
            input_vid_list_tmp.remove(vid)
    input_vid_list = input_vid_list_tmp

    # find videocoding .json files corresponding to videos ; remove from list videos that don't have corresponding json file
    pattern = re.compile(".*_(\d{8}_\d{6}_\d{3})\.mp4")
    videocoding_files = glob.glob(f'{args.inputpath_json}/*/*')
    input_json_list = []
    input_vid_list_tmp = input_vid_list.copy()

    for i, vid in enumerate(input_vid_list):
        timeids = pattern.findall(vid)
        if len(timeids) != 1:
            raise RuntimeError
        timeid = timeids[0]
        count = 0
        for vcf in videocoding_files:
            if timeid in vcf:
                input_json_list.append(vcf)
                count += 1
                break
        if count == 0:
            input_vid_list_tmp.remove(vid)
    input_vid_list = input_vid_list_tmp

    # select randomly 8 videos to process
    idx_list = random.sample(range(len(input_vid_list)), 8)
    vid_list = []
    json_list = []
    for idx in idx_list:
        vid_list.append(input_vid_list[idx])
        json_list.append(input_json_list[idx])
    writejson('prepare_annotations/already_processed_videos.json', already_processed_videos+[v.split('/')[-1] for v in vid_list])

    
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
    
    for vid, jjson in zip(vid_list, json_list):
        print(f'\nProcessing {vid}')
        print(jjson)
        disagreement_dict = {'FP':{}, 'FN':{}}

        length_AI, length_AI_score, length_video, extract_path = extract_lengths(
            jjson, vid, vid.replace('mp4', 'csv'),
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
#                try:
#                    FN_list.append(saved_frames[idx-1])
#                except:
#                    print(f'index {idx-1} out of range')
#                try:
#                    FN_list.append(saved_frames[idx+1])
#                except:
#                    print(f'index {idx+1} out of range')
                    
            # saved frames corresponding to false negatives
            try:
                FN_dict[cls] += frames_list
            except:
                FN_dict[cls] = frames_list

        full_list += FN_list
            
    full_list = list(set(full_list))
    print(len(full_list))

    ########### load AI model for active learning and pre-annotation #####################
    import mmdet.apis
    model = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device=device,
    )

    # active learning to keep only N frames to annotate
    config = mmcv.Config.fromfile(args.config)
    try:
        test_cfg = config.model.bbox_head.test_cfg
    except:
        test_cfg = config.model.test_cfg

    rev_disagreement_dict = revert_dict(FP_dict, FN_dict, full_list) # reverted disagreement dict (key=frames, value=list of classes)
            
    uncertainty = []
    for im in tqdm.tqdm(full_list):
        weight = 0
        for cls in rev_disagreement_dict[im]: # weigh uncertainty by importance of degradation class
            weight += cls_weight[cls]
        try:
            uncertainty.append( mmdet.apis.inference_detector(model, im, active_learning=True)*weight )
        except:
            print(im)
            uncertainty.append(torch.tensor([0], device=device))

    uncertainty = torch.concat(uncertainty)
    selection = select_images(test_cfg.active_learning.selection_method, uncertainty, int(args.n_sel_al), **test_cfg.active_learning.selection_kwargs)
    
    # copy images to annotate and pre-annotate them with detection model
    dt = datetime.datetime.now()
    timestamp = f'{dt.year}{dt.month}{dt.day}_{dt.hour}{dt.minute}'
    outpath = f'prepare_annotations/images_to_annotate_{timestamp}'
    os.makedirs(f'{outpath}/FN/', exist_ok=True)
    os.makedirs(f'{outpath}/FP/', exist_ok=True)

    for idx in tqdm.tqdm(selection):
        im = full_list[idx]

        # find if image is FN
        cls_list = []
        for cls, images in FN_dict.items():
            if im in images:
                cls_list.append(cls)
        
        video_name = im.split('/')[-3]
        im_name = int(float(im.split('/')[-1].replace('.jpg', '')))
        im_path = im.replace(' ', '\ ')
        if len(cls_list) == 0:
            im_name_new = f'{outpath}/FP/{video_name.replace(" ", "")}_{im_name}.jpg'
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
    
    
