import argparse
import json
import os
import re
import glob
import datetime

import cv2 as cv
import numpy as np
import tqdm
import mmcv
from mmdet.utils import select_images
import torch


# weight associated to each class, coming from formula to grade road segments
cls_weight = {
    'Arrachement_pelade' : 5,
    'Faiencage' : 7,
    'Nid_de_poule' : 5,
    'Transversale' : 1.5,
    'Longitudinale' : 6,
    'Remblaiement_de_tranchees' : 2,
    'Autre_reparation' : 2,
}



def revert_dict(FP_dict, FN_dict, im_list):
    rev_dict = {}
    for im in im_list:
        cls_list = []
        im_name = float(im.split('/')[-1].replace('.png', ''))
        for cls, images in FP_dict.items():
            if im_name in images:
                cls_list.append(cls)
                
        for cls, images in FN_dict.items():
            if im in images:
                cls_list.append(cls)
        rev_dict[im] = cls_list
        
    return rev_dict



def main(args):

    timestamp = int(datetime.datetime.now().timestamp())
    os.makedirs(f'prepare_annotations/images_to_annotate_{timestamp}/FN/', exist_ok=True)
    os.makedirs(f'prepare_annotations/images_to_annotate_{timestamp}/FP/', exist_ok=True)

    full_list = []
    FN_dict = {}
    
    for inputpath in args.inputpath:
    
        with open(f'{inputpath}/disagreement_dict.json') as f:
            disagreement_dict = json.load(f)

        # False Positive: simply send images for annotation
        FP_list = []
        for key, val in disagreement_dict['FP'].items():
            FP_list += val

        FP_list = [f'{inputpath}/{im}.png' for im in FP_list]
        full_list += FP_list

        # False Negatives: take saved frame closest to extracted length + surrounding frames
        saved_frames = glob.glob(f'{inputpath}/*png')
        saved_frames = np.sort(np.array([float(sf.replace(inputpath, '').replace('.png', '')) for sf in saved_frames]))

        FN_list = []
        for cls, val in disagreement_dict['FN'].items():
            frames_list = []
            for v in val:
                idx = np.argmin(np.abs(saved_frames - v))
                FN_list.append(f'{inputpath}/{saved_frames[idx]}.png')
                frames_list.append(f'{inputpath}/{saved_frames[idx]}.png')
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

    rev_disagreement_dict = revert_dict(disagreement_dict['FP'], FN_dict, full_list) # reverted disagreement dict (key=frames, value=list of classes)
    
    # load model
    import mmdet.apis
    model = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device='cuda:1',
    )

    # active learning
    config = mmcv.Config.fromfile(args.config)
    try:
        test_cfg = config.model.bbox_head.test_cfg
    except:
        test_cfg = config.model.test_cfg

    uncertainty = []
    for im in full_list:
        weight = 0
        for cls in rev_disagreement_dict[im]:
            weight += cls_weight[cls]
        uncertainty.append( mmdet.apis.inference_detector(model, im, active_learning=True)*weight )

    uncertainty = torch.concat(uncertainty)
    selection = select_images(test_cfg.active_learning.selection_method, uncertainty, test_cfg.active_learning.n_sel, **test_cfg.active_learning.selection_kwargs)

    # copy images to annotate and pre-annotate them with detection model
    for idx in selection:
        im = full_list[idx]

        # find if image is FN
        cls_list = []
        for cls, images in FN_dict.items():
            if im in images:
                cls_list.append(cls)
        
        video_name = im.split('/')[-4]
        im_name = int(float(im.split('/')[-1].replace('.png', '')))
        im_path = im.replace(' ', '\ ')
        if len(cls_list) == 0:
            im_name_new = f'images_to_annotate/FP/{video_name.replace(" ", "")}_{im_name}.png'
        else:
            FN_name = '_'.join(cls_list)
            os.makedirs(f'images_to_annotate/FN/{FN_name}', exist_ok=True)
            im_name_new = f'images_to_annotate/FN/{FN_name}/{video_name.replace(" ", "")}_{im_name}.png'
        os.system(f'cp {im_path} {im_name_new}')

        image = cv.imread(im_name_new)

        ann = []
            
        res = mmdet.apis.inference_detector(model, image)
        image_width = image.shape[1]
        image_height = image.shape[0]
        for ic, c in enumerate(res): # loop on classes
            for ip, p in enumerate(c): # loop on bboxes
                x1, y1, x2, y2 = p[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if p[4] > 0.3:
                    ann.append(f'{ic} {(x1+x2)/2/image_width} {(y1+y2)/2/image_height} {(x2-x1)/image_width} {(y2-y1)/image_height}')
                        
        ann_name = im_name_new.replace('.png', '.txt')
        with open(ann_name, 'w') as f:
            for a in ann:
                f.write(f'{a}\n')
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath', nargs='*', help='path to processed images and disagreement_dict.json produced for a given video with compare_AI_videocoding.py.')
    parser.add_argument('--config', required=True, help='config for detection model.')
    parser.add_argument('--checkpoint', required=True, help='weights for detection model.')
    
    args = parser.parse_args()

    main(args)
    
    
