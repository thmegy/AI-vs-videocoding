import argparse
import json
import os
import re

import cv2 as cv
import numpy as np
import tqdm
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import strip_accents, Haversine_distance, parse_gps_info, parse_videocoding, get_length_timestamp_map



# dict to link classes from videocoding and AI
classes_vid = {
    'Affaissement de rive' : '',
    'Affaissement hors rive' : '',
    'Ornierage' : '',
    'Arrachement' : 'Arrachement_pelade',
    'Faiencage' : 'Faiencage',
    'Fissure longitudinale BDR' : 'Longitudinale',
    'Fissure longitudinale HBDR' : 'Longitudinale',
    'Fissure transversale' : 'Transversale',
    'Nid de poule' : 'Nid_de_poule',
    'Ressuage - Glacage' : '',
    'Reparation en BB sur decoupe' : 'Remblaiement_de_tranchees',
    'Autre reparation' : 'Autre_reparation'
    }

classes_comp = {
    'Arrachement_pelade' : 0,
    'Faiencage' : 1,
    'Nid_de_poule' : 2,
    'Transversale' : 3,
    'Longitudinale' : 4,
    'Remblaiement_de_tranchees' : 5,
    'Autre_reparation' : 6,
}

# dict with pre-defined videos and their corresponding csv and json
inputpath = 'data/cracks/videocodage/'
comp_videos_1 = {
    'CC_BHS_logiroad_2_20210531_161706_390.mp4' : 'releve_lot_DEGRADATION_IA_20220513_075518_PC-GAETAN_AV_CC_BHS_logiroad_2_20210531_161706_390.json',
    'CC_BHS_logiroad_2_20210601_080253_408.mp4' : 'releve_lot_DEGRADATION_IA_20220513_080725_PC-GAETAN_AV_CC_BHS_logiroad_2_20210601_080253_408.json',
    'CC_BHS_logiroad_2_20210707_140609_300.mp4' : 'releve_lot_DEGRADATION_IA_20220516_080820_PC-GAETAN_AV_CC_BHS_logiroad_2_20210707_140609_300.json',
    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_DEGRADATION_IA_20220516_081017_PC-GAETAN_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
    'CD27_EURE_EURE_1_20220209_140240_074.mp4' : 'releve_lot_DEGRADATION_IA_20220516_081508_PC-GAETAN_AV_CD27_EURE_EURE_1_20220209_140240_074.json',
    'CD27_EURE_EURE_1_20220323_082403_092.mp4' : 'releve_lot_DEGRADATION_IA_20220516_082201_PC-GAETAN_AV_CD27_EURE_EURE_1_20220323_082403_092.json',
    'Douarnenez_AV_Logiroad_AV_20220315_120749_017.mp4' : 'releve_lot_DEGRADATION_IA_20220516_083956_PC-GAETAN_AV_Douarnenez_AV_Logiroad_AV_20220315_120749_017.json',
    'Douarnenez_AV_Logiroad_AV_20220317_085336_460.mp4' : 'releve_lot_DEGRADATION_IA_20220516_085227_PC-GAETAN_AV_Douarnenez_AV_Logiroad_AV_20220317_085336_460.json',
    'Douarnenez_AV_Logiroad_AV_20220317_172419_603.mp4' : 'releve_lot_DEGRADATION_IA_20220516_085819_PC-GAETAN_AV_Douarnenez_AV_Logiroad_AV_20220317_172419_603.json',
    'no_contract_CD 33_20210309_101639_889.mp4' : 'releve_lot_DEGRADATION_IA_20220516_091622_PC-GAETAN_AV_no_contract_CD_33_20210309_101639_889.json',
    'no_contract_CD 33_20210311_111829_045.mp4' : 'releve_lot_DEGRADATION_IA_20220516_093808_PC-GAETAN_AV_no_contract_CD_33_20210311_111829_045.json',
    'no_contract_CD 33_20210420_164326_010.mp4' : 'releve_lot_DEGRADATION_IA_20220516_100415_PC-GAETAN_AV_no_contract_CD_33_20210420_164326_010.json',
    'Vannes_logiroad_2_20220124_094528_789.mp4' : 'releve_lot_DEGRADATION_IA_20220516_101245_PC-GAETAN_AV_Vannes_logiroad_2_20220124_094528_789.json',
    'Vannes_logiroad_2_20220124_165314_007.mp4' : 'releve_lot_DEGRADATION_IA_20220516_101835_PC-GAETAN_AV_Vannes_logiroad_2_20220124_165314_007.json',
}

comp_videos_2 = {
    'CC_BHS_logiroad_2_20210531_161706_390.mp4' : 'releve_lot_IA_20220509_095322_MSI-LEO_AV_CC_BHS_logiroad_2_20210531_161706_390.json',
    'CC_BHS_logiroad_2_20210601_080253_408.mp4' : 'releve_lot_IA_20220510_083631_MSI-LEO_AV_CC_BHS_logiroad_2_20210601_080253_408.json',
    'CC_BHS_logiroad_2_20210707_140609_300.mp4' : 'releve_lot_IA_20220510_090913_MSI-LEO_AV_CC_BHS_logiroad_2_20210707_140609_300.json',
    'CD27_EURE_EURE_1_20220112_114308_180.mp4' : 'releve_lot_IA_20220510_091349_MSI-LEO_AV_CD27_EURE_EURE_1_20220112_114308_180.json',
    'CD27_EURE_EURE_1_20220209_140240_074.mp4' : 'releve_lot_IA_20220510_091814_MSI-LEO_AV_CD27_EURE_EURE_1_20220209_140240_074.json',
    'CD27_EURE_EURE_1_20220323_082403_092.mp4' : 'releve_lot_IA_20220510_092425_MSI-LEO_AV_CD27_EURE_EURE_1_20220323_082403_092.json',
    'Douarnenez_AV_Logiroad_AV_20220315_120749_017.mp4' : 'releve_lot_IA_20220510_104854_MSI-LEO_AV_Douarnenez_AV_Logiroad_AV_20220315_120749_017.json',
    'Douarnenez_AV_Logiroad_AV_20220317_085336_460.mp4' : 'releve_lot_IA_20220510_112359_MSI-LEO_AV_Douarnenez_AV_Logiroad_AV_20220317_085336_460.json',
    'Douarnenez_AV_Logiroad_AV_20220317_172419_603.mp4' : 'releve_lot_IA_20220510_113342_MSI-LEO_AV_Douarnenez_AV_Logiroad_AV_20220317_172419_603.json',
    'no_contract_CD 33_20210309_101639_889.mp4' : 'releve_lot_IA_20220510_094310_MSI-LEO_AV_no_contract_CD_33_20210309_101639_889.json',
    'no_contract_CD 33_20210311_111829_045.mp4' : 'releve_lot_IA_20220510_100500_MSI-LEO_AV_no_contract_CD_33_20210311_111829_045.json',
    'no_contract_CD 33_20210420_164326_010.mp4' : 'releve_lot_IA_20220510_102610_MSI-LEO_AV_no_contract_CD_33_20210420_164326_010.json',
    'Vannes_logiroad_2_20220124_094528_789.mp4' : 'releve_lot_IA_20220510_161404_MSI-LEO_AV_Vannes_logiroad_2_20220124_094528_789.json',
    'Vannes_logiroad_2_20220124_165314_007.mp4' : 'releve_lot_IA_20220510_173127_MSI-LEO_AV_Vannes_logiroad_2_20220124_165314_007.json',
#    'Vannes_logiroad_2_20220124_165314_007.mp4' : 'releve_lot_IA_20220516_085556_MSI-LEO_AV_Vannes_logiroad_2_20220124_165314_007.json',
}



def extract_lengths(jsonpath, geoptis_csvpath):
    '''
    Extract length from start of mission for annotations from videocoder.
    In case of a continuous degradations, extract length every n meter.
    '''
    classes, degradations, timestamps = parse_videocoding(jsonpath)

    traj_times_0, distance_for_timestamp = get_length_timestamp_map(geoptis_csvpath)
    classname_to_deg_index = {name: i for i, name in enumerate(classes)}
    deg_index_to_classname = {i: name for name, i in classname_to_deg_index.items()}

    if degradations.shape[0] == 0:
        return 0, 0

    # length list for each AI class
    length_video = [[] for _ in range(len(classes_comp))]

    process_every_nth_meter = 3

    # loop over videocoding annotations --> extract lengths
    for i_timestamp, cur_timestamps in enumerate(timestamps):
        try:
            start, end = distance_for_timestamp(cur_timestamps)
        except:
            continue
        
        # single-frame
        if start == end:
            degradation_indexes = np.nonzero(degradations[i_timestamp])
            if len(degradation_indexes) != 1 and len(degradation_indexes[0]) != 1:
                print("Several degradations in one timestamp")
                exit(0)
            degradation_index = degradation_indexes[0][0]
            degradation_name = deg_index_to_classname[degradation_index]
            degradation_name = strip_accents(degradation_name)
            degradation_name = degradation_name[:-2] # remove letter for name of videocoder (G for Gaetan, L for Leo)
            if classes_vid[degradation_name] != '':
                idx = classes_comp[classes_vid[degradation_name]]
                length_video[idx].append(start)

        # continuous degradations
        if start != end:
            dist_array = np.arange(start, end, process_every_nth_meter) # discretise continuous degradations
            
            degradation_indexes = np.nonzero(degradations[i_timestamp])
            if len(degradation_indexes) != 1 and len(degradation_indexes[0]) != 1:
                print("Several degradations in one timestamp")
                exit(0)
            degradation_index = degradation_indexes[0][0]
            degradation_name = deg_index_to_classname[degradation_index]
            degradation_name = strip_accents(degradation_name)
            degradation_name = degradation_name[:-2] # remove letter for name of videocoder (G fpr Gaetan, L for Leo)
            if classes_vid[degradation_name] != '':
                idx = classes_comp[classes_vid[degradation_name]]
                for d in dist_array:
                    length_video[idx].append(d)

    return length_video



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
                
        
    
def main(args):

    results = {'recall':{}, 'precision':{}}

    outpath = f'plots/videocoders/'
    os.makedirs(outpath, exist_ok=True)
        
    length_video_list_1 = []
    length_video_list_2 = []

    # get length for videocoding and predictions from all videos
    for vid, jjson in comp_videos_1.items():
        length_video = extract_lengths(inputpath+'/'+jjson, inputpath+'/'+vid.replace('mp4', 'csv'))
        length_video_list_1.append(length_video)
    for vid, jjson in comp_videos_2.items():
        length_video = extract_lengths(inputpath+'/'+jjson, inputpath+'/'+vid.replace('mp4', 'csv'))
        length_video_list_2.append(length_video)

    # compute distances for all classes                
    # loop over classes
    for ic in range(len(classes_comp)):
        class_name = list(classes_comp.keys())[ic]
        results['recall'][class_name] = {}
        results['precision'][class_name] = {}
        
        distances_video_list_1 = []
        distances_video_list_2 = []
            
        # loop over videos
        for length_video_1, length_video_2 in zip(length_video_list_1, length_video_list_2):
            lv1 = length_video_1[ic]
            lv2 = length_video_2[ic]

            distances_video_list_1.append( compute_smallest_distances(lv1, lv2) )
            distances_video_list_2.append( compute_smallest_distances(lv2, lv1) )
            
        distances_video_1 = np.concatenate(distances_video_list_1)
        distances_video_2 = np.concatenate(distances_video_list_2)

        # plot distance distributions
        plot_distance_distributions(distances_video_1, distances_video_2, outpath, class_name)
                
        # compute precision and recall
        # number of true positives is different when taken from distances from videocoding or from AI prediction, e.g. one videocoding matches wit several predictions...
        recall_dict = {}
        precision_dict = {}
        for dthr in args.threshold:
            precision, recall = compute_precision_recall(distances_video_2, distances_video_1, dthr)
            recall_dict[f'{int(dthr)}m'] = recall
            precision_dict[f'{int(dthr)}m'] = precision

        results['recall'][class_name] = recall_dict
        results['precision'][class_name] = precision_dict


        plot_evolution(results, 'precision', 'Precision', outpath)
        plot_evolution(results, 'recall', 'Recall', outpath)
        
        with open(f'{outpath}/results.json', 'w') as fout:
            json.dump(results, fout, indent = 6)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, nargs='*', default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                        help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')
    args = parser.parse_args()

    main(args)
    
