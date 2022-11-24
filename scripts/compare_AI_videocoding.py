import argparse
import json, hjson
import os
import re
import glob

import cv2 as cv
import numpy as np
import tqdm
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import strip_accents, Haversine_distance, parse_gps_info, parse_videocoding, get_length_timestamp_map, extract_lengths


def compute_smallest_distances(length_1, length_2):
    # compute smallest distance for each element of length_1 wrt length_2 (arrays or lists)
    distances = []        
    for l1 in length_1:
        if len(length_2) > 0:
            dist = np.abs(l1 - np.array(length_2))
            smallest_dist = np.min(dist)
        else:
            smallest_dist = 50
        distances.append(smallest_dist)
    return np.array(distances)



def compute_average_precision(distances_AI, score, threshold):
    # sort examples
    sort_inds = np.argsort(-score)
    sort_dist = distances_AI[sort_inds]
    
    # count true positive examples
    pos_inds = sort_dist < threshold
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]
    
    # count not difficult examples
    pn_inds = sort_dist != -1
    pn = np.cumsum(pn_inds)
    
    tp[np.logical_not(pos_inds)] = 0
    precision = tp / pn
    ap = np.sum(precision) / total_pos

    return ap



def compute_precision_recall(distances_AI, distances_video, threshold):
    tp_ai = (distances_AI < threshold).sum()
    fp = (distances_AI > threshold).sum()
    
    tp_video = (distances_video < threshold).sum()
    fn = (distances_video > threshold).sum()
    
    recall = tp_video / (tp_video + fn)
    precision = tp_ai / (tp_ai + fp)

    return precision, recall



def plot_distance_distributions(distances_AI, distances_video, outpath, class_name, thr):
    bins = np.linspace(0, 50, 11)
    
    fig, ax1 = plt.subplots()
    ax1.hist(np.clip(distances_AI, bins[0], bins[-1]), bins=bins)
    ax1.set_xlabel('distance [m]')
    fig.savefig(f'{outpath}/{class_name}_AI_thr0{int(thr*10)}.png') # videocoding as reference
    plt.close()
                
    fig, ax1 = plt.subplots()
    ax1.hist(np.clip(distances_video, bins[0], bins[-1]), bins=bins)
    ax1.set_xlabel('distance [m]')
    fig.savefig(f'{outpath}/{class_name}_video_thr0{int(thr*10)}.png') # AI as reference
    plt.close()


def plot_precision_recall(results, dthr, score_thresholds, outpath, classes_comp):
    fig_precision, axp = plt.subplots() # precision summary plot
    axp.set_xlabel('score threshold')
    axp.set_ylabel('precision')
    fig_recall, axr = plt.subplots() # recall summary plot
    axr.set_xlabel('score threshold')
    axr.set_ylabel('recall')
    fig_precision_recall, axpr = plt.subplots() # precision-recall summary plot
    axpr.set_xlabel('recall')
    axpr.set_ylabel('precision')

    for ic in range(len(classes_comp)):
        class_name = list(classes_comp.keys())[ic]
        precision_list = []
        recall_list = []
        for thr in score_thresholds:
            precision_list.append(results['precision'][class_name][thr][f'{int(dthr)}m'])
            recall_list.append(results['recall'][class_name][thr][f'{int(dthr)}m'])

        axp.plot(score_thresholds, precision_list, label=class_name)
        axr.plot(score_thresholds, recall_list, label=class_name)
        axpr.plot(recall_list, precision_list, label=class_name, marker='o')

    axp.legend()
    fig_precision.savefig(f'{outpath}/precision_{int(dthr)}m.png') # AI as reference
    axr.legend()
    fig_recall.savefig(f'{outpath}/recall_{int(dthr)}m.png') # AI as reference
    axpr.legend()
    fig_precision_recall.savefig(f'{outpath}/precision_recall_{int(dthr)}m.png') # AI as reference
    plt.close('all')
                
        
    
def main(args):

    # load dicts to link classes from AI and videocoding
    cls_config =  hjson.load(open(args.cls_config, 'r'))
    classes_vid = cls_config['classes_vid']
    classes_AI = cls_config['classes_AI']
    classes_comp= cls_config['classes_comp']
        
    results = {'recall':{}, 'precision':{}, 'ap':{}}
    if args.type=='seg': # not controlling scores for segmentation
        thrs = [0.]
    else:
        thrs = np.linspace(0.3, 0.8, 6)

    # no input video given, treat all predefined videos and combine results
    if args.inputvideo is None:
        # load dicts to link predefined videos and their videocoding files
        videocoding_config =  hjson.load(open(args.videocoding_config, 'r'))
        json_dict = videocoding_config['json_dict']
        inputpath = videocoding_config['inputpath']
        
        outpath = f'results_comparison/{args.type}/'
        os.makedirs(outpath, exist_ok=True)
        
        length_AI_list = []
        length_AI_score_list = []
        length_video_list = []

        # get length for videocoding and predictions from all videos
        for vid, jjson in json_dict.items():
            print(f'\n{vid}')
            length_AI, length_AI_score, length_video, extract_path = extract_lengths(
                f'{inputpath}/{jjson}', f'{inputpath}/{vid}', f'{inputpath}/{vid.replace("mp4", "csv")}',
                classes_vid, classes_AI, classes_comp,
                args.type, args.config, args.checkpoint, args.process_every_nth_meter, filter_road=args.filter_road
            )

            length_AI_list.append(length_AI)
            length_AI_score_list.append(length_AI_score)
            length_video_list.append(length_video)

        # compute distances for all combinations of classes and thresholds                
        # loop over classes
        for ic in range(len(classes_comp)):
            class_name = list(classes_comp.keys())[ic]
            results['recall'][class_name] = {}
            results['precision'][class_name] = {}

            distances_AI_list = [[] for _ in range(len(thrs))]
            distances_video_list = [[] for _ in range(len(thrs))]
            
            distances_AI_full = [] # for AP
            score_full = [] # for AP
            
            # loop over videos
            for length_AI, length_AI_score, length_video in zip(length_AI_list, length_AI_score_list, length_video_list):
                lai = np.array(length_AI[ic])
                score = np.array(length_AI_score[ic])
                lv = length_video[ic]

                distances_AI_full += compute_smallest_distances(lai, lv).tolist()
                score_full += length_AI_score[ic]

                # loop over thresholds (0.3 -> 0.8)
                for it, thr in enumerate(thrs):
                    lai_thr = lai[score > thr]  # apply threshold to length_AI

                    distances_AI_list[it].append( compute_smallest_distances(lai_thr, lv) )
                    distances_video_list[it].append( compute_smallest_distances(lv, lai_thr) )

            
            ######## compute average precision ##########
            if args.type != 'seg':
                distances_AI_full = np.array(distances_AI_full)
                score_full = np.array(score_full)
                ap_dict = {}
                for dthr in args.threshold_dist:
                    ap = compute_average_precision(distances_AI_full, score_full, dthr)
                    ap_dict[f'{int(dthr)}m'] = ap
                results['ap'][class_name] = ap_dict

            
            # new loop over thresholds to compute precision and recall across all videos
            for it, thr in enumerate(thrs):
                distances_AI = np.concatenate(distances_AI_list[it])
                distances_video = np.concatenate(distances_video_list[it])

                # plot distance distributions
                plot_distance_distributions(distances_AI, distances_video, outpath, class_name, thr)
                
                # compute precision and recall
                # number of true positives is different when taken from distances from videocoding or from AI prediction, e.g. one videocoding matches wit several predictions...
                recall_dict = {}
                precision_dict = {}
                for dthr in args.threshold_dist:
                    precision, recall = compute_precision_recall(distances_AI, distances_video, dthr)
                    recall_dict[f'{int(dthr)}m'] = recall
                    precision_dict[f'{int(dthr)}m'] = precision

                results['recall'][class_name][thr] = recall_dict
                results['precision'][class_name][thr] = precision_dict

        with open(f'{outpath}/results.json', 'w') as fout:
            json.dump(results, fout, indent = 6)

        for dthr in args.threshold_dist:
            plot_precision_recall(results, dthr, thrs, outpath, classes_comp)

            
    # treat a single video
    else:
        if args.extract_disagreement:
            thrs = [args.threshold_score]
            disagreement_dict = {'FP':{}, 'FN':{}}
            
        outpath = f'results_comparison/{args.type}/{args.inputvideo.split("/")[-1].replace(".mp4", "")}'
        os.makedirs(outpath, exist_ok=True)
            
        if args.csv is None:
            args.csv = args.inputvideo.replace('mp4', 'csv')
        
        if args.csv != args.inputvideo.replace('mp4', 'csv'):
            csv_path = args.csv.split('/')
            csv_path = '/'.join(csv_path[:-1])
            df = pd.read_csv(args.csv, delimiter=';', header=None, quotechar='%')
            for vid in pd.unique(df[0]):
                df[df[0]==vid].to_csv(f'{csv_path}/{vid}.csv', header=False, sep=';', index=False, quotechar='%')
            args.csv = args.inputvideo.replace('mp4', 'csv')

        if args.json is None:
            video_name = args.inputvideo.split('/')[-1]
            video_path = '/'.join(args.inputvideo.split('/')[:-1])
            args.json = video_path+'/'+comp_videos[video_name]

        length_AI, length_AI_score, length_video, extract_path = extract_lengths(
            args.json, args.inputvideo, args.csv,
            classes_vid, classes_AI, classes_comp,
            args.type, args.config, args.checkpoint, args.process_every_nth_meter, filter_road=args.filter_road
        )

        # loop over classes
        for ic, (lai, score, lv) in enumerate(zip(length_AI, length_AI_score, length_video)):
            class_name = list(classes_comp.keys())[ic]
            results['recall'][class_name] = {}
            results['precision'][class_name] = {}
            
            lai = np.array(lai)
            score = np.array(score)
            
            
            ########## compute average precision
            if args.type != 'seg':
                distances_AI_full = compute_smallest_distances(lai, lv) # distances with no score threshold applied
                ap_dict = {}
                for dthr in args.threshold_dist:
                    ap = compute_average_precision(distances_AI_full, score, dthr)
                    ap_dict[f'{int(dthr)}m'] = ap
                results['ap'][class_name] = ap_dict

            
            # loop over thresholds
            for thr in thrs:
                lai_thr = lai[score > thr]  # apply threshold to length_AI

                distances_AI = compute_smallest_distances(lai_thr, lv)
                distances_video = compute_smallest_distances(lv, lai_thr)

                # extract frames with disagreement between AI and videocoding
                if args.extract_disagreement:
                    # AI predictions but no videocoding annotations (FP)
                    disagreement_dict['FP'][class_name] = np.sort(lai_thr[distances_AI>10]).tolist()
                    # videocoding annotations but no AI predictions (FN)
                    disagreement_dict['FN'][class_name] = np.sort(np.array(lv)[distances_video>10]).tolist()

                # plot
                plot_distance_distributions(distances_AI, distances_video, outpath, class_name, thr)

                # compute precision and recall
                # number of true positives is different when taken from distances from videocoding or from AI prediction, e.g. one videocoding matches wit several predictions...
                recall_dict = {}
                precision_dict = {}
                for dthr in args.threshold_dist:
                    precision, recall = compute_precision_recall(distances_AI, distances_video, dthr)
                    recall_dict[f'{int(dthr)}m'] = recall
                    precision_dict[f'{int(dthr)}m'] = precision

                results['recall'][class_name][thr] = recall_dict
                results['precision'][class_name][thr] = precision_dict

        with open(f'{outpath}/results.json', 'w') as fout:
            json.dump(results, fout, indent = 6)

        if args.extract_disagreement:
            with open(f'{extract_path}/disagreement_dict.json', 'w') as fout:
                json.dump(disagreement_dict, fout, indent = 6)
        else:# do not plot metrics in disagreement extraction mode
            for dthr in args.threshold_dist:
                plot_precision_recall(results, dthr, thrs, outpath, classes_comp)
            

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # args for single videos
    parser.add_argument('--inputvideo')
    parser.add_argument('--json')
    parser.add_argument('--csv')
    
    # args for AI
    parser.add_argument('--type', choices=['cls', 'det', 'seg'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    
    parser.add_argument('--cls-config', default='configs/classes_reference_videos.json', help='json file with dicts to link classes from AI and videocoding.')
    parser.add_argument('--videocoding-config', default='configs/videocoding_reference_videos.json', help='json file with dicts to videocoding files corresponding to predefined videos.')
    
    parser.add_argument('--process-every-nth-meter', type=float, default=3, help='step in meters between processed frames.')
    parser.add_argument('--threshold-dist', type=float, nargs='*', default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                        help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')

    parser.add_argument('--filter-road', action='store_true', help='Apply segmentation model to keep only road pixels on images.')

    parser.add_argument('--extract-disagreement', action='store_true', help='Extract frames where AI and videocoding disagree (distance > 10m).')
    parser.add_argument('--threshold-score', type=float, default=0.3, help='score threshold to be used for disagreement extraction.')
    
    args = parser.parse_args()

    main(args)
    
