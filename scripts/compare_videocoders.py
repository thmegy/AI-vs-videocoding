import argparse
import json, hjson
import os
import re

import cv2 as cv
import numpy as np
import tqdm
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import extract_lengths_videocoding


# dict with pre-defined videos and their corresponding csv and json
inputpath = 'data/reference_videos/'
#comp_videos_1 = {
#    'CC_BHS_logiroad_2_20210531_161706_390.mp4' : 'releve_lot_DEGRADATION_IA_20220513_075518_PC-GAETAN_AV_CC_BHS_logiroad_2_20210531_161706_390.json',
#    'CC_BHS_logiroad_2_20210601_080253_408.mp4' : 'releve_lot_DEGRADATION_IA_20220513_080725_PC-GAETAN_AV_CC_BHS_logiroad_2_20210601_080253_408.json',
#    'CC_BHS_logiroad_2_20210707_140609_300.mp4' : 'releve_lot_DEGRADATION_IA_20220516_080820_PC-GAETAN_AV_CC_BHS_logiroad_2_20210707_140609_300.json',
#    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_DEGRADATION_IA_20220516_081017_PC-GAETAN_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
#    'CD27_EURE_EURE_1_20220209_140240_074.mp4' : 'releve_lot_DEGRADATION_IA_20220516_081508_PC-GAETAN_AV_CD27_EURE_EURE_1_20220209_140240_074.json',
#    'CD27_EURE_EURE_1_20220323_082403_092.mp4' : 'releve_lot_DEGRADATION_IA_20220516_082201_PC-GAETAN_AV_CD27_EURE_EURE_1_20220323_082403_092.json',
#    'Douarnenez_AV_Logiroad_AV_20220315_120749_017.mp4' : 'releve_lot_DEGRADATION_IA_20220516_083956_PC-GAETAN_AV_Douarnenez_AV_Logiroad_AV_20220315_120749_017.json',
#    'Douarnenez_AV_Logiroad_AV_20220317_085336_460.mp4' : 'releve_lot_DEGRADATION_IA_20220516_085227_PC-GAETAN_AV_Douarnenez_AV_Logiroad_AV_20220317_085336_460.json',
#    'Douarnenez_AV_Logiroad_AV_20220317_172419_603.mp4' : 'releve_lot_DEGRADATION_IA_20220516_085819_PC-GAETAN_AV_Douarnenez_AV_Logiroad_AV_20220317_172419_603.json',
##    'no_contract_CD 33_20210309_101639_889.mp4' : 'releve_lot_DEGRADATION_IA_20220516_091622_PC-GAETAN_AV_no_contract_CD_33_20210309_101639_889.json',
#    'no_contract_CD 33_20210311_111829_045.mp4' : 'releve_lot_DEGRADATION_IA_20220516_093808_PC-GAETAN_AV_no_contract_CD_33_20210311_111829_045.json',
#    'no_contract_CD 33_20210420_164326_010.mp4' : 'releve_lot_DEGRADATION_IA_20220516_100415_PC-GAETAN_AV_no_contract_CD_33_20210420_164326_010.json',
#    'Vannes_logiroad_2_20220124_094528_789.mp4' : 'releve_lot_DEGRADATION_IA_20220516_101245_PC-GAETAN_AV_Vannes_logiroad_2_20220124_094528_789.json',
#    'Vannes_logiroad_2_20220124_165314_007.mp4' : 'releve_lot_DEGRADATION_IA_20220516_101835_PC-GAETAN_AV_Vannes_logiroad_2_20220124_165314_007.json',
#}

comp_videos_1 = {
    'CC_BHS_logiroad_2_20210531_161706_390.mp4' : 'releve_lot_IA_20220509_095322_MSI-LEO_AV_CC_BHS_logiroad_2_20210531_161706_390.json',
    'CC_BHS_logiroad_2_20210601_080253_408.mp4' : 'releve_lot_IA_20220510_083631_MSI-LEO_AV_CC_BHS_logiroad_2_20210601_080253_408.json',
    'CC_BHS_logiroad_2_20210707_140609_300.mp4' : 'releve_lot_IA_20220510_090913_MSI-LEO_AV_CC_BHS_logiroad_2_20210707_140609_300.json',
    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_IA_20220510_091349_MSI-LEO_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
    'CD27_EURE_EURE_1_20220209_140240_074.mp4' : 'releve_lot_IA_20220510_091814_MSI-LEO_AV_CD27_EURE_EURE_1_20220209_140240_074.json',
    'CD27_EURE_EURE_1_20220323_082403_092.mp4' : 'releve_lot_IA_20220510_092425_MSI-LEO_AV_CD27_EURE_EURE_1_20220323_082403_092.json',
    'Douarnenez_AV_Logiroad_AV_20220315_120749_017.mp4' : 'releve_lot_IA_20220510_104854_MSI-LEO_AV_Douarnenez_AV_Logiroad_AV_20220315_120749_017.json',
    'Douarnenez_AV_Logiroad_AV_20220317_085336_460.mp4' : 'releve_lot_IA_20220510_112359_MSI-LEO_AV_Douarnenez_AV_Logiroad_AV_20220317_085336_460.json',
    'Douarnenez_AV_Logiroad_AV_20220317_172419_603.mp4' : 'releve_lot_IA_20220510_113342_MSI-LEO_AV_Douarnenez_AV_Logiroad_AV_20220317_172419_603.json',
#    'no_contract_CD 33_20210309_101639_889.mp4' : 'releve_lot_IA_20220510_094310_MSI-LEO_AV_no_contract_CD_33_20210309_101639_889.json',
    'no_contract_CD 33_20210311_111829_045.mp4' : 'releve_lot_IA_20220510_100500_MSI-LEO_AV_no_contract_CD_33_20210311_111829_045.json',
    'no_contract_CD 33_20210420_164326_010.mp4' : 'releve_lot_IA_20220510_102610_MSI-LEO_AV_no_contract_CD_33_20210420_164326_010.json',
    'Vannes_logiroad_2_20220124_094528_789.mp4' : 'releve_lot_IA_20220510_161404_MSI-LEO_AV_Vannes_logiroad_2_20220124_094528_789.json',
#    'Vannes_logiroad_2_20220124_165314_007.mp4' : 'releve_lot_IA_20220510_173127_MSI-LEO_AV_Vannes_logiroad_2_20220124_165314_007.json',
    'Vannes_logiroad_2_20220124_165314_007.mp4' : 'releve_lot_IA_20220516_085556_MSI-LEO_AV_Vannes_logiroad_2_20220124_165314_007.json',
}

# Nestor
comp_videos_2 = {
    'CC_BHS_logiroad_2_20210531_161706_390.mp4' : 'releve_lot_Comparaison_IA_20220606_101729_LAPTOP-HO492LF6_AV_CC_BHS_logiroad_2_20210531_161706_390.json',
#    'CC_BHS_logiroad_2_20210531_161706_390.mp4' : 'releve_lot_Comparaison_IA_20220606_135312_LAPTOP-HO492LF6_AV_CC_BHS_logiroad_2_20210531_161706_390.json',
    'CC_BHS_logiroad_2_20210601_080253_408.mp4' : 'releve_lot_Comparaison_IA_20220608_154238_LAPTOP-HO492LF6_AV_CC_BHS_logiroad_2_20210601_080253_408.json',
    'CC_BHS_logiroad_2_20210707_140609_300.mp4' : 'releve_lot_Comparaison_IA_20220611_132026_LAPTOP-HO492LF6_AV_CC_BHS_logiroad_2_20210707_140609_300.json',
    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_Comparaison_IA_20220523_153632_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
#    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_Comparaison_IA_20220524_084808_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
#    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_Comparaison_IA_20220601_162221_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
#    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_Comparaison_IA_20220603_105427_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
#    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_Comparaison_IA_20220606_135553_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
    'CD27_EURE_EURE_1_20220209_140240_074.mp4' : 'releve_lot_Comparaison_IA_20220523_230624_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220209_140240_074.json',
#    'CD27_EURE_EURE_1_20220209_140240_074.mp4' : 'releve_lot_Comparaison_IA_20220606_124838_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220209_140240_074.json',
    'CD27_EURE_EURE_1_20220323_082403_092.mp4' : 'releve_lot_Comparaison_IA_20220523_094116_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220323_082403_092.json',
#    'CD27_EURE_EURE_1_20220323_082403_092.mp4' : 'releve_lot_Comparaison_IA_20220606_141658_LAPTOP-HO492LF6_AV_CD27_EURE_EURE_1_20220323_082403_092.json',
    'Douarnenez_AV_Logiroad_AV_20220315_120749_017.mp4' : 'releve_lot_Comparaison_IA_20220530_204909_LAPTOP-HO492LF6_AV_Douarnenez_AV_Logiroad_AV_20220315_120749_017.json',
#    'Douarnenez_AV_Logiroad_AV_20220315_120749_017.mp4' : 'releve_lot_Comparaison_IA_20220601_161803_LAPTOP-HO492LF6_AV_Douarnenez_AV_Logiroad_AV_20220315_120749_017.json',
#    'Douarnenez_AV_Logiroad_AV_20220315_120749_017.mp4' : 'releve_lot_Comparaison_IA_20220603_105843_LAPTOP-HO492LF6_AV_Douarnenez_AV_Logiroad_AV_20220315_120749_017.json',
    'Douarnenez_AV_Logiroad_AV_20220317_085336_460.mp4' : 'releve_lot_Comparaison_IA_20220611_212617_LAPTOP-HO492LF6_AV_Douarnenez_AV_Logiroad_AV_20220317_085336_460.json',
#    'Douarnenez_AV_Logiroad_AV_20220317_085336_460.mp4' : 'releve_lot_Comparaison_IA_20220611_234222_LAPTOP-HO492LF6_AV_Douarnenez_AV_Logiroad_AV_20220317_085336_460.json',
    'Douarnenez_AV_Logiroad_AV_20220317_172419_603.mp4' : 'releve_lot_Comparaison_IA_20220523_161056_LAPTOP-HO492LF6_AV_Douarnenez_AV_Logiroad_AV_20220317_172419_603.json',
#    'no_contract_CD 33_20210309_101639_889.mp4' : '',
    'no_contract_CD 33_20210311_111829_045.mp4' : 'releve_lot_Comparaison_IA_20220611_231623_LAPTOP-HO492LF6_AV_no_contract_CD_33_20210311_111829_045.json',
    'no_contract_CD 33_20210420_164326_010.mp4' : 'releve_lot_Comparaison_IA_20220601_130547_LAPTOP-HO492LF6_AV_no_contract_CD_33_20210420_164326_010.json',
    'Vannes_logiroad_2_20220124_094528_789.mp4' : 'releve_lot_Comparaison_IA_20220610_155307_LAPTOP-HO492LF6_AV_Vannes_logiroad_2_20220124_094528_789.json',
    'Vannes_logiroad_2_20220124_165314_007.mp4' : 'releve_lot_Comparaison_IA_20220610_124821_LAPTOP-HO492LF6_AV_Vannes_logiroad_2_20220124_165314_007.json',
}




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



def compute_precision_recall(distances_AI, distances_video, threshold):
    tp_ai = (distances_AI < threshold).sum()
    fp = (distances_AI > threshold).sum()
    
    tp_video = (distances_video < threshold).sum()
    fn = (distances_video > threshold).sum()
    
    recall = tp_video / (tp_video + fn)
    precision = tp_ai / (tp_ai + fp)

    return precision, recall



def plot_distance_distributions(distances_video_1, distances_video_2, outpath, class_name):
    bins = np.linspace(0, 50, 11)
    
    fig, ax1 = plt.subplots()
    ax1.hist(np.clip(distances_video_1, bins[0], bins[-1]), bins=bins)
    ax1.set_xlabel('distance [m]')
    fig.savefig(f'{outpath}/{class_name}_videocoder_1.png') # videocoder 2 as reference
    plt.close()
                
    fig, ax1 = plt.subplots()
    ax1.hist(np.clip(distances_video_2, bins[0], bins[-1]), bins=bins)
    ax1.set_xlabel('distance [m]')
    fig.savefig(f'{outpath}/{class_name}_videocoder_2.png') # videocoder 1 as reference
    plt.close()

    

def plot_evolution(res, var, var_name, outpath):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlabel('threshold [m]', fontsize=14)
    ax.set_ylabel(var_name, fontsize=14)

    for cls, var_dict in res[var].items():
        thrs = [t.replace('m', '') for t in var_dict.keys()]
        ax.plot(thrs, var_dict.values(), label=cls)
    
    ax.legend(fontsize=16, ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.savefig(f'{outpath}/{var}.png') # videocoder 1 as reference
    plt.close()


    
def plot_precision_recall(results, outpath):
    fig, ax = plt.subplots() # precision-recall summary plot
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')

    for ic, class_name in enumerate(results['recall'].keys()):
        precision_list = list(results['precision'][class_name].values())
        recall_list = list(results['recall'][class_name].values())
        distance_thrs = list(results['recall'][class_name].keys())

        p = ax.plot(recall_list, precision_list, label=class_name, marker='o')
        col = p[-1].get_color()
        for x, y, dthr in zip(recall_list, precision_list, distance_thrs):
            plt.text(x, y, dthr.replace('m', ''), color=col)

    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig(f'{outpath}/precision_recall.png')
    plt.close('all')
    
                
        
    
def main(args):
    # load dicts with classes in json and classes used for comparaison
    cls_config =  hjson.load(open(args.cls_config, 'r'))
    classes_vid = cls_config['classes_vid']
    classes_comp= cls_config['classes_comp']

    # load dicts to link predefined videos and their videocoding files
    videocoding_config =  hjson.load(open(args.videocoding_config, 'r'))
    json_dict_list = [ videocoding_config['json_dict_gaetan'], videocoding_config['json_dict_leo'], videocoding_config['json_dict_nestor'] ]
    inputpath = videocoding_config['inputpath']

    
    # get length for each videocoder from all videos
    length_video_list_videocoders = [] #(N_videocoders, N_video, N_classes, N_degradations)
    for json_dict in json_dict_list:
        length_video_list = [] #(N_videos, N_classes, N_degradations)
        for vid, jjson in json_dict.items():
            length_video = extract_lengths_videocoding(inputpath+'/'+jjson, inputpath+'/'+vid.replace('mp4', 'csv'),
                                                       classes_vid, classes_comp, args.process_every_nth_meter)
            length_video_list.append(length_video)
            
        length_video_list_videocoders.append(length_video_list)

        
    # loop over videocoders taken as reference
    for ic, (length_video_list_ref, videocoder_ref) in enumerate(zip(length_video_list_videocoders, args.videocoders)):
        # remove reference videocoders from lists
        length_video_list_videocoders_tmp = length_video_list_videocoders[:ic] + length_video_list_videocoders[ic+1:]
        videocoders_tmp = args.videocoders[:ic] + args.videocoders[ic+1:]

        # loop over the other videocoders
        for length_video_list_other, videocoder_other in zip(length_video_list_videocoders_tmp, videocoders_tmp):
    
            results = {'recall':{}, 'precision':{}}
            
            outpath = f'results_comparison/videocoders/{videocoder_ref}_vs_{videocoder_other}'
            os.makedirs(outpath, exist_ok=True)
        

            # compute distances for all classes          
            # loop over classes
            for ic in range(len(classes_comp)):
                class_name = list(classes_comp.keys())[ic]
                results['recall'][class_name] = {}
                results['precision'][class_name] = {}
        
                distances_video_list_ref = []
                distances_video_list_other = []
            
                # loop over videos
                for length_video_ref, length_video_other in zip(length_video_list_ref, length_video_list_other):
                    lv1 = length_video_ref[ic]
                    lv2 = length_video_other[ic]

                    distances_video_list_ref.append( compute_smallest_distances(lv1, lv2) )
                    distances_video_list_other.append( compute_smallest_distances(lv2, lv1) )
            
                distances_video_ref = np.concatenate(distances_video_list_ref)
                distances_video_other = np.concatenate(distances_video_list_other)

                # plot distance distributions
                plot_distance_distributions(distances_video_ref, distances_video_other, outpath, class_name)
                
                # compute precision and recall
                # number of true positives is different when taken from distances from videocoding or from AI prediction, e.g. one videocoding matches wit several predictions...
                recall_dict = {}
                precision_dict = {}
                for dthr in args.threshold:
                    precision, recall = compute_precision_recall(distances_video_other, distances_video_ref, dthr)
                    recall_dict[f'{int(dthr)}m'] = recall
                    precision_dict[f'{int(dthr)}m'] = precision

                results['recall'][class_name] = recall_dict
                results['precision'][class_name] = precision_dict


                plot_evolution(results, 'precision', 'Precision', outpath)
                plot_evolution(results, 'recall', 'Recall', outpath)
                plot_precision_recall(results, outpath)
        
                with open(f'{outpath}/results.json', 'w') as fout:
                    json.dump(results, fout, indent = 6)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videocoders', nargs=3, type=str, default=['Gaetan', 'Leo', 'Nestor'], help='name of videocoders we compare AI to.')
    parser.add_argument('--videocoding-config', default='configs/videocoding_reference_videos.json', help='json file with dicts to videocoding files corresponding to predefined videos.')
    parser.add_argument('--cls-config', default='configs/classes_reference_videos.json', help='json file with dicts to link classes from AI and videocoding.')
    
    parser.add_argument('--process-every-nth-meter', type=float, default=3, help='step in meters between processed frames.')
    parser.add_argument("--threshold", type=float, nargs='*', default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                        help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')
    args = parser.parse_args()

    main(args)
    
