import argparse
import json, hjson
import os
import re
import glob

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import extract_lengths_videocoding, extract_lengths_AI, compute_smallest_distances, compute_distances


def compute_average_precision(distances_AI, score, threshold_dist):
    # sort examples
    sort_inds = np.argsort(-score)
    sort_dist = distances_AI[sort_inds]
    
    # count true positive examples
    pos_inds = sort_dist < threshold_dist
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]
    
    # count not difficult examples
    pn_inds = sort_dist != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / pn
    ap = np.sum(precision) / total_pos

    return ap, precision[pos_inds], score[sort_inds][pos_inds], # AP, [precision_array], [score of true positives]


def compute_average_recall(distance_array, score_array, dist_thr, score_thrs):
    if len(distance_array) == 0 or len(score_thrs) == 0:
        return np.nan, []

    # loop in score thresholds, at each iteration compute number of FN
    recall_list = []
    for score_thr in score_thrs:
        masked_distance = np.where(score_array > score_thr, distance_array, 50) # set distances corresponding to scores below score threshold to 50m --> above dist threshold
        distance_thresholded = np.where(masked_distance<dist_thr, 1, 0) # replace distances below dist thr by 1, other by 0
        TP_count = distance_thresholded.sum(axis=1) # count of AI detection within detection threshold of each annotated degradation
        recall = (TP_count > 0).sum() / len(TP_count) # TP / (TP+FN)
        recall_list.append(recall)

    ar = sum(recall_list) / len(recall_list)
    return ar, recall_list

        

def plot_precision_recall(results, dthr, outpath):
    figpr, axpr = plt.subplots() # precision-recall summary plot
    axpr.set_xlabel('recall')
    axpr.set_ylabel('precision')

    figf1, axf1 = plt.subplots(figsize=(8,6)) # F1-score summary plot
    axf1.set_xlabel('score threshold')
    axf1.set_ylabel('F1-score')

    for cls, cls_dict in results.items():
        dthr_dict = cls_dict[dthr]
        
        sthr_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        for sthr, sthr_dict in dthr_dict.items():
            if sthr == 'ap' or sthr == 'ar':
                continue
            sthr_list.append(float(sthr))
            precision_list.append(sthr_dict['precision'])
            recall_list.append(sthr_dict['recall'])
            f1_list.append(sthr_dict['f1_score'])

        axpr.plot(recall_list, precision_list, label=cls, marker='o', markersize=4)
        p = axf1.plot(sthr_list, f1_list, label=cls)
        col = p[-1].get_color()

        # get max f1 score and add to plot
        maxf1_id = np.argmax(np.array(f1_list))
        xmax = np.array(sthr_list)[maxf1_id]
        ymax = np.array(f1_list)[maxf1_id]
        axf1.plot([xmax, xmax], [0, ymax], color=col, linestyle='--', linewidth=1)
        axf1.plot([0, xmax], [ymax, ymax], color=col, linestyle='--', linewidth=1)
        plt.text(xmax, 0, f'{xmax:.2f}', color=col, horizontalalignment='right', verticalalignment='top', rotation=45, fontsize='small')
        plt.text(0, ymax, f'{ymax:.2f}', color=col, horizontalalignment='right', verticalalignment='center', fontsize='small')

    axpr.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize='small')
    figpr.set_tight_layout(True)
    figpr.savefig(f'{outpath}/precision_recall_{dthr}.png')

    axf1.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize='medium')
    axf1.set_xlim(0)
    axf1.set_ylim(0)
    figf1.set_tight_layout(True)
    figf1.savefig(f'{outpath}/f1_score_{dthr}.png')

    plt.close('all')
                
        
    
def main(args):

    # load dicts to link classes from AI and videocoding
    cls_config =  hjson.load(open(args.cls_config, 'r'))
    classes_vid = cls_config['classes_vid']
    classes_AI = cls_config['classes_AI']
    classes_comp= cls_config['classes_comp']
        
    results = {}

    # number of detection to keep for average recall calculation =
    # number of frames in interval given by +-
    # the largest distance threshold considered (detections beyand are necessarily FN)
    N_ai = 2 * (1 + args.threshold_dist[-1] // args.process_every_nth_meter)

    # no input video given, treat all predefined videos and combine results
    if args.inputvideo is None:
        outpath_base = f'results_comparison/{args.type}'
        
        if not args.post_process:
            # load dicts to link predefined videos and their videocoding files
            videocoding_config =  hjson.load(open(args.videocoding_config, 'r'))
            json_dict_list = [ videocoding_config['json_dict_gaetan'], videocoding_config['json_dict_leo'], videocoding_config['json_dict_nestor'] ]
            inputpath = videocoding_config['inputpath']

            # get length for videocoding (for all videocoders) and predictions from all videos
            length_AI_list = []
            length_AI_score_list = []
            for vid in json_dict_list[0].keys():
                print(f'\n{vid}')
                length_AI, length_AI_score, extract_path = extract_lengths_AI(
                    f'{inputpath}/{vid}', f'{inputpath}/{vid.replace("mp4", "csv")}', classes_AI, classes_comp,
                    args.type, args.config, args.checkpoint, args.process_every_nth_meter, filter_road=args.filter_road
                )
                length_AI_list.append(length_AI)
                length_AI_score_list.append(length_AI_score)


            length_video_list_videocoders = [] # (N_videocoders, N_videos, N_classes, N_degradations)
            for json_dict in json_dict_list:
                length_video_list = [] # (N_videos, N_classes, N_degradations)
                for vid in json_dict.keys():
                    jfile = json_dict[vid]
                    length_video = extract_lengths_videocoding(
                        f'{inputpath}/{jfile}', f'{inputpath}/{vid.replace("mp4", "csv")}',
                        classes_vid, classes_comp, args.process_every_nth_meter
                    )

                    length_video_list.append(length_video)

                length_video_list_videocoders.append(length_video_list)


            # compute distances for all combinations of classes and thresholds                
            # loop over videocoders
            for length_video_list, videocoder in zip(length_video_list_videocoders, args.videocoders):
                outpath = f'{outpath_base}/{videocoder}'
                os.makedirs(outpath, exist_ok=True)

                # loop over classes
                for ic in range(len(classes_comp)):
                    class_name = list(classes_comp.keys())[ic]

                    distances_AI_full = [] # for AP, dim N_detection_AI
                    score_full = [] # for AP, dim N_detection_AI
                    
                    distances_array_full = [] # for AR, dim (N_degradation_vid, N_ai)
                    score_array_full = [] # for AR, dim (N_degradation_vid, N_ai)

                    # loop over videos
                    for length_AI, length_AI_score, length_video in zip(length_AI_list, length_AI_score_list, length_video_list):
                        lai = np.array(length_AI[ic])
                        score = np.array(length_AI_score[ic])
                        lv = length_video[ic]

                        distances_AI_full += compute_smallest_distances(lai, lv).tolist()
                        score_full += length_AI_score[ic]
                        
                        if len(lv) > 0 and len(lai) > 0:
                            distances_array, score_array = compute_distances(lv, lai, score, N_ai) # for average recall
                            distances_array_full.append(distances_array)
                            score_array_full.append(score_array)

                    # combine results of videos
                    distances_AI_full = np.array(distances_AI_full)
                    score_full = np.array(score_full)
                    if len(distances_array_full) > 0 and len(score_array_full) > 0:
                        distances_array_full = np.concatenate(distances_array_full)
                        score_array_full = np.concatenate(score_array_full)
                    else:
                        distances_AI_array_full, score_array_full = np.array([]), np.array([])

                    ap_dict = {}
                    ar_dict = {}
                    precision_recall_dict = {}
                    for dthr in args.threshold_dist:
                        ap, precision_list, tp_scores = compute_average_precision(distances_AI_full, score_full, dthr)
                        ap_dict[f'{int(dthr)}m'] = ap

                        ar, recall_list = compute_average_recall(distances_array_full, score_array_full, dthr, tp_scores)
                        ar_dict[f'{int(dthr)}m'] = ar

                        precision_recall_dict[f'{int(dthr)}m'] = {'ap':ap, 'ar':ar}
                        for p, r, s in zip(precision_list, recall_list, tp_scores):
                            precision_recall_dict[f'{int(dthr)}m'][f'{s:.3f}'] = {'precision':p, 'recall':r, 'f1_score':2*p*r/(p+r)}

                    results[class_name] = precision_recall_dict

                with open(f'{outpath}/results.json', 'w') as fout:
                    json.dump(results, fout, indent = 6)

            
    # treat a single video
    else:            
        outpath_base = f'results_comparison/{args.type}/{args.inputvideo.split("/")[-1].replace(".mp4", "")}'

        if not args.post_process:
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
                # load dicts to link predefined videos and their videocoding files
                videocoding_config =  hjson.load(open(args.videocoding_config, 'r'))
                json_dict_list = [ videocoding_config['json_dict_gaetan'], videocoding_config['json_dict_leo'], videocoding_config['json_dict_nestor'] ]

                video_name = args.inputvideo.split('/')[-1]
                video_path = '/'.join(args.inputvideo.split('/')[:-1])
                args.json = [ f'{video_path}/{json_dict[video_name]}' for json_dict in json_dict_list ]

            # extract images and run inference
            length_AI, length_AI_score, extract_path = extract_lengths_AI(
                args.inputvideo, args.csv, classes_AI, classes_comp,
                args.type, args.config, args.checkpoint, args.process_every_nth_meter, filter_road=args.filter_road
                )

            # loop over videocoders
            for jfile, videocoder in zip(args.json, args.videocoders):
                outpath = f'{outpath_base}/{videocoder}'
                os.makedirs(outpath, exist_ok=True)

                length_video = extract_lengths_videocoding(
                    jfile, args.csv, classes_vid,
                    classes_comp, args.process_every_nth_meter
                )

                # loop over classes
                for ic, (lai, score, lv) in enumerate(zip(length_AI, length_AI_score, length_video)):
                    class_name = list(classes_comp.keys())[ic]
                    results[class_name] = {}

                    lai = np.array(lai)
                    score = np.array(score)

                    distances_AI_full = compute_smallest_distances(lai, lv) # distances with no score threshold applied
                    if len(lv) > 0 and len(lai) > 0:
                        distances_AI_array_full, score_array_full = compute_distances(lv, lai, score, N_ai) # for average recall
                    else:
                        distances_AI_array_full, score_array_full = np.array([]), np.array([])

                    ap_dict = {}
                    ar_dict = {}
                    precision_recall_dict = {}
                    for dthr in args.threshold_dist:
                        ap, precision_list, tp_scores = compute_average_precision(distances_AI_full, score, dthr)
                        ap_dict[f'{int(dthr)}m'] = ap

                        ar, recall_list = compute_average_recall(distances_AI_array_full, score_array_full, dthr, tp_scores)
                        ar_dict[f'{int(dthr)}m'] = ar

                        precision_recall_dict[f'{int(dthr)}m'] = {'ap' : ap, 'ar' : ar}
                        for p, r, s in zip(precision_list, recall_list, tp_scores):
                            precision_recall_dict[f'{int(dthr)}m'][f'{s:.3f}'] = {'precision':p, 'recall':r, 'f1_score':2*p*r/(p+r)}

                    results[class_name] = precision_recall_dict

                with open(f'{outpath}/results.json', 'w') as fout:
                    json.dump(results, fout, indent = 6)

                    
    ######### post-processing ##############
    distance_thr = '10m' # threshold used to extract single summary figures
    score_thrs = cls_config['score_thresholds']

    for videocoder in args.videocoders:
        print(f'\n\n{videocoder}\n')
        outpath = f'{outpath_base}/{videocoder}'
        with open(f'{outpath}/results.json', 'r') as f:
            results = json.load(f)

        for dthr in args.threshold_dist:
            plot_precision_recall(results, f'{int(dthr)}m', outpath)

        print('class  AP  F1_score')
#        for (cls, ap_dict), f1_dict in zip(results['ap'].items(), results['f1_score'].values()):
#            print(cls, f'{ap_dict[distance_thr]:.3f}', f'{f1_dict[score_thrs[cls]][distance_thr]:.3f}')


            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--videocoders', nargs=3, type=str, default=['Gaetan', 'Leo', 'Nestor'], help='name of videocoders we compare AI to.')
    parser.add_argument('--process-every-nth-meter', type=float, default=3, help='step in meters between processed frames.')
    
    # args for single videos
    parser.add_argument('--inputvideo')
    parser.add_argument('--json', nargs=3, type=str, help='path to json files from Gaetan, Leo and Nestor (in the order of "--videocoders").')
    parser.add_argument('--csv')
    
    # args for AI
    parser.add_argument('--type', choices=['cls', 'det'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    
    parser.add_argument('--cls-config', default='configs/classes_reference_videos.json', help='json file with dicts to link classes from AI and videocoding.')
    parser.add_argument('--videocoding-config', default='configs/videocoding_reference_videos.json', help='json file with dicts to videocoding files corresponding to predefined videos.')
    
    parser.add_argument('--threshold-dist', type=float, nargs='*', default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                        help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')

    parser.add_argument('--filter-road', action='store_true', help='Apply segmentation model to keep only road pixels on images.')

    # post-precessing
    parser.add_argument('--post-process', action='store_true', help='Make summary plots without processing videos again.')
    
    args = parser.parse_args()

    main(args)
    
