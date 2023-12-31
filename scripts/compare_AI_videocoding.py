import argparse
import json, hjson
import os
import re
import glob
import sys

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import extract_lengths_videocoding, extract_lengths_AI, compute_smallest_distances, compute_distances



def compute_average_precision(distances_AI, score, threshold_dist):
    '''
    Compute precision for every true positive detection, and corresponding average precision.
    '''
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



def compute_average_precision_multiref(distances_list, score, threshold_dist):
    '''
    Compute precision for every true positive detection, and corresponding average precision, 
    considering the combination of several videocoders as reference.
    '''
    # get number of videocoders agreeing with each IA detection
    N_vid_agree = (distances_list < threshold_dist).sum(axis=0)
    consensus_threshold = int(distances_list.shape[0] / 2 - 1)

    # sort examples
    sort_inds = np.argsort(-score)
    sort_det = N_vid_agree[sort_inds]

    sorted_score = score[sort_inds]
    # count true positive examples
    pos_inds = sort_det > consensus_threshold
    pos_weighted = np.where(pos_inds, sort_det, 0) # weight by number of agreeing videocoders
    tp = np.cumsum(pos_weighted)
    total_pos = pos_inds.sum() # number of considered score thresholds
    
    # count total examples
    N_videocoders = distances_list.shape[0]
    tot_weighted = np.where(pos_weighted==0, N_videocoders-sort_det , pos_weighted)
    tot = np.cumsum(tot_weighted)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / tot
    ap = np.sum(precision) / total_pos

    return ap, precision[pos_inds], score[sort_inds][pos_inds], tp[pos_inds], tot[pos_inds] # AP, [precision_array], [score of true positives]



def compute_average_recall(distance_array, score_array, dist_thr, score_thrs):
    '''
    Compute recall for every threshold in <score_thrs>, and corresponding average recall.
    Inputs:
    distance_array: (N_degradation_annot, N_ai)
    score_array: (N_degradation_annot, N_ai)
    '''
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



def compute_average_recall_multiref(distance_array_list, score_array_list, distances_videocoders, dist_thr, score_thrs):
    '''
    Compute recall for every threshold in <score_thrs>, and corresponding average recal,
    considering the combination of several videocoders as reference.
    Inputs:
    - distance_array_list: smallest distances between annotations of videocoders and AI detections (N_videocoders, N_degradation_annot, N_ai)
    - score_array_list: (N_videocoders, N_degradation_annot, N_ai)
    - distances_videocoders (list[np.array]): list of 2D arrays giving for the annotations of each videocoder the smallest distance to an annotation 
    of every other videocoder(N_videocoder, (N_videocoder-1, N_degradations))     
    '''
#    if len(distance_array) == 0 or len(score_thrs) == 0:
#        return np.nan, []

    # for every annotations of a given videocoder, get number of agreeing videocoders
    N_vid_agree_list = []
    for dist in distances_videocoders:
        N_vid_agree = (dist < dist_thr).sum(axis=0)
        N_vid_agree_list.append(N_vid_agree+1)
    consensus_threshold = int(len(distance_array_list) / 2 - 1)

    # loop in score thresholds, at each iteration compute number of FN
    recall_list = []
    for score_thr in score_thrs:
        # loop over videocoders
        TP_count_tot = 0
        TP_FN_count_tot = 0 # TP+FN
        for distance_array, score_array, N_vid_agree in zip(distance_array_list, score_array_list, N_vid_agree_list):
            # consider only annotations with at least 2 agreeing videocoders
            if len(score_array) > 0:
                distance_array = distance_array[N_vid_agree>consensus_threshold]
                score_array = score_array[N_vid_agree>consensus_threshold]

                masked_distance = np.where(score_array > score_thr, distance_array, 50) # set distances corresponding to scores below score threshold to 50m --> above dist threshold
                distance_thresholded = np.where(masked_distance<dist_thr, 1, 0) # replace distances below dist thr by 1, other by 0
                TP_count = distance_thresholded.sum(axis=1) # count of AI detection within detection threshold of each annotated degradation
                TP_count_tot += (TP_count > 0).sum()
                TP_FN_count_tot += len(TP_count)
                
            
        recall = TP_count_tot / TP_FN_count_tot # TP / (TP+FN)
        recall_list.append(recall)

    try:
        ar = sum(recall_list) / len(recall_list)
    except:
        ar = 0.
        
    return ar, recall_list



def compute_distances_videocoders(length_video_list_videocoders, cls_index):
    '''
    Compute distances between one videocoder and all the others.
    Inputs:
    - length_video_list_videocoders: list of annotations for each class, video, and videocoder. Dimensions = (N_videocoders, N_videos, N_classes, N_degradations).
    - cls_index: index of consodered class.
    Returns:
    - distances_videocoders (list[np.array]): list of 2D arrays giving for the annotations of each videocoder the smallest distance to an annotation 
    of every other videocoder(N_videocoder, (N_videocoder-1, N_degradations))
    '''

    distances_videocoders = []
    
    # loop over videocoders
    for videocoder_comp_id, length_video_list_comp in enumerate(length_video_list_videocoders):
        length_video_list_videocoders_ref = length_video_list_videocoders[:videocoder_comp_id] + length_video_list_videocoders[videocoder_comp_id+1:]
        # compare videocoder_comp to all the others
        # loop over reference videocoders
        distances_videocoder_comp = []
        for length_video_list_ref in length_video_list_videocoders_ref:
            # loop over videos
            distances = []
            for iv, length_video in enumerate(length_video_list_ref):
                lv_comp = length_video_list_comp[iv][cls_index]
                lv_ref = length_video[cls_index]

                distances += compute_smallest_distances(lv_comp, lv_ref).tolist()

            distances_videocoder_comp.append( distances )

        distances_videocoders.append( np.stack(distances_videocoder_comp) )

    return distances_videocoders

        

def plot_precision_recall(results, dthr, outpath):
    figpr, axpr = plt.subplots() # precision-recall summary plot
    axpr.set_xlabel('recall')
    axpr.set_ylabel('precision')

    figf1, axf1 = plt.subplots(figsize=(8,6)) # F1-score summary plot
    axf1.set_xlabel('score threshold')
    axf1.set_ylabel('F1-score')

    max_f1_dict = {}
    max_sthr_dict = {}
    
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
        if len(f1_list) > 0:
            maxf1_id = np.argmax(np.array(f1_list))
            xmax = np.array(sthr_list)[maxf1_id]
            ymax = np.array(f1_list)[maxf1_id]
            axf1.plot([xmax, xmax], [0, ymax], color=col, linestyle='--', linewidth=1)
            axf1.plot([0, xmax], [ymax, ymax], color=col, linestyle='--', linewidth=1)
            plt.text(xmax, 0, f'{xmax:.2f}', color=col, horizontalalignment='right', verticalalignment='top', rotation=45, fontsize='small')
            plt.text(0, ymax, f'{ymax:.2f}', color=col, horizontalalignment='right', verticalalignment='center', fontsize='small')
        else:
            xmax, ymax = np.nan, np.nan
        max_f1_dict[cls] = ymax
        max_sthr_dict[cls] = xmax

    axpr.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize='small')
    figpr.set_tight_layout(True)
    figpr.savefig(f'{outpath}/precision_recall_{dthr}.png')

    axf1.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize='medium')
    axf1.set_xlim(0)
    axf1.set_ylim(0)
    figf1.set_tight_layout(True)
    figf1.savefig(f'{outpath}/f1_score_{dthr}.png')

    plt.close('all')

    return max_f1_dict, max_sthr_dict
                
        
    
def main(args):

    # load dicts to link classes from AI and videocoding
    cls_config =  hjson.load(open(args.cls_config, 'r'))
    classes_vid = cls_config['classes_vid']
    classes_AI = cls_config['classes_AI']
    classes_comp= cls_config['classes_comp']
        
    # number of detection to keep for average recall calculation =
    # number of frames in interval given by +-
    # the largest distance threshold considered (detections beyand are necessarily FN)
    N_ai = 2 * (1 + args.threshold_dist[-1] // args.process_every_nth_meter)

#            if args.csv != args.inputvideo.replace('mp4', 'csv'):
#                csv_path = args.csv.split('/')
#                csv_path = '/'.join(csv_path[:-1])
#                df = pd.read_csv(args.csv, delimiter=';', header=None, quotechar='%')
#                for vid in pd.unique(df[0]):
#                    df[df[0]==vid].to_csv(f'{csv_path}/{vid}.csv', header=False, sep=';', index=False, quotechar='%')
#                args.csv = args.inputvideo.replace('mp4', 'csv')


    outpath_base = f'results_comparison/{args.type}'

    if not args.post_process:
        # load dicts to link predefined videos and their videocoding files
        videocoding_config =  hjson.load(open(args.videocoding_config, 'r'))
        json_dict_list = []
        for iv, videocoder in enumerate(args.videocoders):
            json_dict = videocoding_config[f'json_dict_{videocoder}'] # load dict to link predefined videos and their videocoding files
            json_dict_list.append(json_dict)
        inputpath = videocoding_config['inputpath']

        # get length for videocoding (for all videocoders) and predictions from all videos
        length_AI_list = []
        length_AI_score_list = []
        for vid in json_dict_list[0].keys():
            print(f'\n{vid}')
            length_AI, length_AI_score, extract_path = extract_lengths_AI(
                f'{inputpath}/{vid}', f'{inputpath}/{vid.replace("mp4", "csv")}', classes_AI, classes_comp,
                args.type, args.config, args.checkpoint, args.process_every_nth_meter
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

        distances_AI_videocoders_list = [[] for _ in range(len(classes_comp))] # for AP combining all videocoders (N_classes, N_videocoders, N_detections)
        AI_scores = [] # for AP combining all videocoders (N_classes, N_detections)
        distances_array_videocoders_list = [[] for _ in range(len(classes_comp))] # for AR combining all videocoders (N_classes, N_videocoders, N_degradation_annot, N_ai)
        AI_scores_array = [[] for _ in range(len(classes_comp))] # for AR combining all videocoders (N_classes, N_videocoders, N_detections, N_ai)

        for iv, (length_video_list, videocoder) in enumerate(zip(length_video_list_videocoders, args.videocoders)):
            results = {}

            outpath = f'{outpath_base}/{videocoder}'
            os.makedirs(outpath, exist_ok=True)

            # loop over classes
            for ic in range(len(classes_comp)):
                class_name = list(classes_comp.keys())[ic]

                distances_AI_full = [] # for AP, dim N_detection_AI
                score_full = [] # for AP, dim N_detection_AI

                distances_array_full = [] # for AR, dim (N_degradation_annot, N_ai)
                score_array_full = [] # for AR, dim (N_degradation_annot, N_ai)

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
                distances_AI_videocoders_list[ic].append(distances_AI_full)
                if iv==0:
                    AI_scores.append(score_full)
                if len(distances_array_full) > 0 and len(score_array_full) > 0:
                    distances_array_full = np.concatenate(distances_array_full)
                    score_array_full = np.concatenate(score_array_full)
                else:
                    distances_array_full, score_array_full = np.array([]), np.array([])
                distances_array_videocoders_list[ic].append(distances_array_full)
                AI_scores_array[ic].append(score_array_full)

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

                
        # AP and AR for combination of videocoders 
        outpath = f'{outpath_base}/Combination'
        os.makedirs(outpath, exist_ok=True)

        results = {}

        for ic, cls in enumerate(classes_comp):
            class_name = list(classes_comp.keys())[ic]

            distances_videocoders = compute_distances_videocoders(length_video_list_videocoders, ic) # for AR

            ap_dict = {}
            ar_dict = {}
            precision_recall_dict = {}
            for dthr in args.threshold_dist:
                ap, precision_list, tp_scores, n_tp, n_tot = compute_average_precision_multiref(np.stack(distances_AI_videocoders_list[ic]), AI_scores[ic], dthr)
                ap_dict[f'{int(dthr)}m'] = ap

                ar, recall_list = compute_average_recall_multiref(distances_array_videocoders_list[ic], AI_scores_array[ic], distances_videocoders, dthr, tp_scores)
                ar_dict[f'{int(dthr)}m'] = ar

                precision_recall_dict[f'{int(dthr)}m'] = {'ap':ap, 'ar':ar}
                for p, r, s, ntp, ntot in zip(precision_list, recall_list, tp_scores, n_tp, n_tot):
                    precision_recall_dict[f'{int(dthr)}m'][f'{s:.5f}'] = {'precision':p, 'recall':r, 'f1_score':2*p*r/(p+r), 'n_tp':float(ntp), 'n_tot':float(ntot)}

            results[class_name] = precision_recall_dict

            
        with open(f'{outpath}/results.json', 'w') as fout:
            json.dump(results, fout, indent = 6)
            


                                
    ######### post-processing ##############
    distance_thr = '10m' # threshold used to extract single summary figures

    for videocoder in args.videocoders + ['Combination']:
        print(f'\n\n{videocoder}\n')
        outpath = f'{outpath_base}/{videocoder}'
        with open(f'{outpath}/results.json', 'r') as f:
            results = json.load(f)

        for dthr in args.threshold_dist:
            max_f1_dict_tmp, max_sthr_dict_tmp = plot_precision_recall(results, f'{int(dthr)}m', outpath)
            if f'{int(dthr)}m' == distance_thr:
                max_f1_dict = max_f1_dict_tmp.copy()
                max_sthr_dict = max_sthr_dict_tmp.copy()

        print(f'{"class" : <30}{"score_thr": ^10}{"AP": ^10}{"AR": ^10}{"F1_score": ^10}')
        for cls, cls_dict in results.items():
            dthr_dict = cls_dict[distance_thr]
            max_sthr = max_sthr_dict[cls]
            max_f1 = max_f1_dict[cls]
            print(f'{cls: <30}{max_sthr: ^10.2f}{dthr_dict["ap"]: ^10.3f}{dthr_dict["ar"]: ^10.3f}{max_f1: ^10.3f}')

        

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--videocoders', nargs='*', type=str, default=['1', '2', '3', '4a', '4b', '5'], help='name of videocoders we compare AI to.')
    parser.add_argument('--process-every-nth-meter', type=float, default=3, help='step in meters between processed frames.')
    
    # args for AI
    parser.add_argument('--type', choices=['cls', 'det'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    
    parser.add_argument('--cls-config', default='configs/classes_reference_videos.json', help='json file with dicts to link classes from AI and videocoding.')
    parser.add_argument('--videocoding-config', default='configs/videocoding_reference_videos.json', help='json file with dicts to videocoding files corresponding to predefined videos.')
    
    parser.add_argument('--threshold-dist', type=float, nargs='*', default=[6, 8, 10, 12, 14, 16, 18, 20],
                        help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')


    # post-precessing
    parser.add_argument('--post-process', action='store_true', help='Make summary plots without processing videos again.')
    
    args = parser.parse_args()

    main(args)
    
