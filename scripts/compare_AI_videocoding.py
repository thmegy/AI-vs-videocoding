import argparse
import json
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

from utils import strip_accents, Haversine_distance, parse_gps_info, parse_videocoding, get_length_timestamp_map



# dict to link classes from videocoding and AI to classes used for comparison
classes_vid = {
    'Affaissement de rive G significatif' : '',
    'Affaissement de rive G grave' : '',
    
    'Affaissement hors rive G significatif' : '',
    'Affaissement hors rive G grave' : '',
    
    'Ornierage G significatif' : '',
    'Ornierage G grave' : '',
    
    'Arrachement G significatif' : 'Arrachement_pelade',
    'Arrachement G grave' : 'Arrachement_pelade',
    
    'Faiencage G significatif non BDR' : 'Faiencage',
    'Faiencage G grave non BDR' : 'Faiencage',
    'Faiencage G specifique BDR' : 'Faiencage',
    
    'Fissure longitudinale BDR G reparee' : 'Pontage',
    'Fissure longitudinale BDR G significative' : 'Longitudinale',
    'Fissure longitudinale BDR G grave' : 'Longitudinale',
    
    'Fissure longitudinale HBDR G reparee' : 'Pontage',
    'Fissure longitudinale HBDR G significative' : 'Longitudinale',
    'Fissure longitudinale HBDR G grave' : 'Longitudinale',
    
    'Fissure transversale G reparee' : 'Pontage',
    'Fissure transversale G significative' : 'Transversale',
    'Fissure transversale G grave' : 'Transversale',
    
    'Nid de poule G significatif' : 'Nid_de_poule',
    'Nid de poule G grave' : 'Nid_de_poule',
    
    'Ressuage - Glacage G localise' : '',
    'Ressuage - Glacage G generalise' : '',
    
    'Reparation en BB sur decoupe G Petite largeur' : 'Remblaiement_de_tranchees',
    'Reparation en BB sur decoupe G Pleine largeur' : 'Remblaiement_de_tranchees',
    
    'Autre reparation G Petite largeur' : 'Autre_reparation',
    'Autre reparation G Pleine largeur' : 'Autre_reparation'
    }

classes_AI = {
    'Arrachement_pelade' : 'Arrachement_pelade',
    'Faiencage' : 'Faiencage',
    'Nid_de_poule' : 'Nid_de_poule',
    'Transversale' : 'Transversale',
    'Longitudinale' : 'Longitudinale',
    'Pontage_de_fissures' : 'Pontage',
    'Remblaiement_de_tranchees' : 'Remblaiement_de_tranchees',
    'Raccord_de_chaussee' : '',
    'Comblage_de_trou_ou_Projection_d_enrobe' : 'Autre_reparation',
    'Bouche_a_clef' : '',
    'Grille_avaloir' : '',
    'Regard_tampon' : ''
}


# classes used for comparison
classes_comp = {
    'Arrachement_pelade' : 0,
    'Faiencage' : 1,
    'Nid_de_poule' : 2,
    'Transversale' : 3,
    'Longitudinale' : 4,
    'Remblaiement_de_tranchees' : 5,
    'Pontage' : 6,
    'Autre_reparation' : 7,
}


# dict with pre-defined videos and their corresponding csv and json
inputpath = 'data/videocodage/'
comp_videos = {
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



def extract_lengths(jsonpath, videopath, geoptis_csvpath):
    '''
    Extract a frame of the video every n meter.
    Run inference on extracted images with pretrained model.
    Return length from start of mission for videocding and predictions.
    '''
    classes, degradations, timestamps = parse_videocoding(jsonpath)

    traj_times_0, distance_for_timestamp = get_length_timestamp_map(geoptis_csvpath)
    classname_to_deg_index = {name: i for i, name in enumerate(classes)}
    deg_index_to_classname = {i: name for name, i in classname_to_deg_index.items()}

    if degradations.shape[0] == 0:
        return 0, 0

    cam = cv.VideoCapture(videopath)
    framerate = cam.get(cv.CAP_PROP_FPS)

    # length list for each AI class
    length_AI = [[] for _ in range(len(classes_comp))]
    length_AI_score = [[] for _ in range(len(classes_comp))] # score associated to prediction
    length_video = [[] for _ in range(len(classes_comp))]

    process_every_nth_meter = args.process_every_nth_meter

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
            if classes_vid[degradation_name] != '':
                idx = classes_comp[classes_vid[degradation_name]]
                for d in dist_array:
                    length_video[idx].append(d)

    vid_name = videopath.split('/')[-1].replace(".mp4","")
    extract_path = f'prepare_annotations/processed_videos/{vid_name}/{process_every_nth_meter}'
    
    if not os.path.isdir(extract_path):
        os.makedirs(extract_path)
        # loop over video
        t = traj_times_0
        d_init = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                break
        
            t += 1 / framerate
            try:
                d = distance_for_timestamp(t)
            except:
                continue
        
            if (d-d_init) < process_every_nth_meter:
                continue

            d_init = d

            # save frame
            cv.imwrite(f'{extract_path}/{d}.png', frame)


    extracted_frames = glob.glob(f'{extract_path}/*.png')

    # road segmentation
    if args.filter_road:
        extract_path += '_road_filter'
        if not os.path.isdir(extract_path):
            os.makedirs(extract_path)
            
            from mmseg.apis import inference_segmentor, init_segmentor
            road_filter = init_segmentor(
                '/home/theo/workdir/mmseg/mmsegmentation/configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes.py',
                '/home/theo/workdir/mmseg/checkpoints/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth',
                device=f'cuda:1'
            )

            for fname in extracted_frames:
                frame = cv.imread(fname)

                frame_seg = inference_segmentor(road_filter, frame)[0]
                mask_road = frame_seg[:,:,np.newaxis] == 0 # road pixels
                frame = np.where(mask_road, frame, 0)

                # save frame
                im_name = fname.split('/')[-1]
                cv.imwrite(f'{extract_path}/{im_name}', frame)

        extracted_frames = glob.glob(f'{extract_path}/*.png')

                
#    ann_path = f'{extract_path}/det_inference_thr_03/'
#    os.makedirs(ann_path, exist_ok=True)

    # load model
    if args.type == 'cls':
        import mmcls.apis
        model = mmcls.apis.init_model(
            args.config,
            args.checkpoint,
            device='cuda:1',
        )
    elif args.type == 'det':
        import mmdet.apis
        model = mmdet.apis.init_detector(
            args.config,
            args.checkpoint,
            device='cuda:1',
        )
    elif args.type == 'seg':
        import mmseg.apis
        model = mmseg.apis.init_segmentor(
            args.config,
            args.checkpoint,
            device='cuda:1',
        )

    # run inference on extracted frames
    for fname in extracted_frames:
        d = float(fname.split('/')[-1].replace('.png', ''))
        frame = cv.imread(fname)
        
        if args.type == 'cls':
            res = mmcls.apis.inference_model(model, frame, is_multi_label=True, threshold=0.01)
            for pc, ps in zip(res['pred_class'], res['pred_score']):
                if classes_AI[pc] != '':
                    idx = classes_comp[classes_AI[pc]]
                    length_AI[idx].append(d)
                    length_AI_score[idx].append(ps)

        elif args.type == 'det':
            ann = []
            
            res = mmdet.apis.inference_detector(model, frame)
            image_width = frame.shape[1]
            image_height = frame.shape[0]
            for ic, c in enumerate(res): # loop on classes
                if (c[:,4] > 0.01).sum() > 0:

                    degradation_name = list(classes_AI.items())[ic][1]
                    if degradation_name != '':
                        idx = classes_comp[degradation_name]
            
                        length_AI[idx].append(d)
                        length_AI_score[idx].append(c[:,4].max()) # take highest score if several instances of same class

#                for ip, p in enumerate(c): # loop on bboxes
#                    x1, y1, x2, y2 = p[:4]
#                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                    if p[4] > 0.3:
#                        ann.append(f'{ic} {(x1+x2)/2/image_width} {(y1+y2)/2/image_height} {(x2-x1)/image_width} {(y2-y1)/image_height}')
#                        
#            ann_name = fname.replace('.png', '.txt').replace(extract_path, ann_path)
#            with open(ann_name, 'w') as f:
#                for a in ann:
#                    f.write(f'{a}\n')

        elif args.type == 'seg':
            res = mmseg.apis.inference_segmentor(model, frame)[0]
            unique_ic, unique_count = np.unique(res, return_counts=True)
            for ic, count in zip(unique_ic, unique_count):
                if ic == 0: # skip background
                    continue
                if count < 800: # keep only degradations big enough
                    continue
                
                degradation_name = list(classes_AI.items())[ic-1][1] # ic-1 to ignore background
                if degradation_name != '':
                    idx = classes_comp[degradation_name]
                    
                    length_AI[idx].append(d)
                    length_AI_score[idx].append(1) # dummy score
            
    return length_AI, length_AI_score, length_video, extract_path



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


def plot_precision_recall(results, dthr, score_thresholds, outpath):
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

    results = {'recall':{}, 'precision':{}, 'ap':{}}
    if args.type=='seg': # not controlling scores for segmentation
        thrs = [0.]
    else:
        thrs = np.linspace(0.3, 0.8, 6)

    # no input video given, treat all predefined videos and combine results
    if args.inputvideo is None:
        outpath = f'plots/{args.type}/'
        os.makedirs(outpath, exist_ok=True)
        
        length_AI_list = []
        length_AI_score_list = []
        length_video_list = []

        # get length for videocoding and predictions from all videos
        for vid, jjson in comp_videos.items():
            print(f'\n{vid}')
            length_AI, length_AI_score, length_video, extract_path = extract_lengths(inputpath+'/'+jjson, inputpath+'/'+vid, inputpath+'/'+vid.replace('mp4', 'csv'))

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
            plot_precision_recall(results, dthr, thrs, outpath)

            
    # treat a single video
    else:
        if args.extract_disagreement:
            thrs = [args.threshold_score]
            disagreement_dict = {'FP':{}, 'FN':{}}
            
        outpath = f'plots/{args.type}/{args.inputvideo.split("/")[-1].replace(".mp4", "")}'
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

        length_AI, length_AI_score, length_video, extract_path = extract_lengths(args.json, args.inputvideo, args.csv)

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
                plot_precision_recall(results, dthr, thrs, outpath)
            

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputvideo')
    parser.add_argument('--json')
    parser.add_argument('--csv')

    parser.add_argument('--type', choices=['cls', 'det', 'seg'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--process-every-nth-meter', type=float, default=3, help='step in meters between processed frames.')
    parser.add_argument('--threshold-dist', type=float, nargs='*', default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                        help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')

    parser.add_argument('--filter-road', action='store_true', help='Apply segmentation model to keep only road pixels on images.')

    parser.add_argument('--extract-disagreement', action='store_true', help='Extract frames where AI and videocoding disagree (distance > 10m).')
    parser.add_argument('--threshold-score', type=float, default=0.3, help='score threshold to be used for disagreement extraction.')
    
    args = parser.parse_args()

    main(args)
    
