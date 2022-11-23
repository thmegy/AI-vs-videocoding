import mmcls.apis
import mmdet.apis
import mmseg.apis
import argparse
import csv
import datetime
import json
import os
import re
import unicodedata

import cv2 as cv
import numpy as np
import tqdm
from scipy.interpolate import interp1d
import pandas as pd



# dict to link classes from videocoding and AI
classes_dict = {
    'Affaissement de rive G' : '',
    'Affaissement hors rive G' : '',
    'Ornierage G' : '',
    'Arrachement G' : 'Arrachement_pelade',
    'Faiencage G' : 'Faiencage',
    'Fissure longitudinale BDR G' : 'Longitudinale',
    'Fissure longitudinale HBDR G' : 'Longitudinale',
    'Fissure transversale G' : 'Transversale',
    'Nid de poule G' : 'Nid_de_poule',
    'Ressuage - Glacage G' : '',
    'Reparation en BB sur decoupe G' : 'Reparation',
    'Autre reparation G' : 'Reparation'
    }

classes_AI = {
    'Arrachement_pelade' : 0,
    'Faiencage' : 1,
    'Nid_de_poule' : 2,
    'Transversale' : 3,
    'Longitudinale' : 4,
    'Reparation' : 5,
}

# dict with pre-defined videos and their corresponding csv and json
inputpath = 'data/cracks/videocodage/'
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



def strip_accents(s):
    # SEE: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )



def Haversine_distance(lnglats1, lnglats2):
    """ Harversine distance in meters.

    SEE: https://en.wikipedia.org/wiki/Haversine_formula
    """
    lnglats1 = np.radians(lnglats1)
    lnglats2 = np.radians(lnglats2)
    lngs1, lats1 = lnglats1[:, 0], lnglats1[:, 1]
    lngs2, lats2 = lnglats2[:, 0], lnglats2[:, 1]
    earth_radius_france = 6366000
    return (
        2
        * earth_radius_france
        * np.arcsin(
            np.sqrt(
                np.sin((lats2 - lats1) / 2.0) ** 2
                + np.cos(lats1) * np.cos(lats2) * np.sin((lngs2 - lngs1) / 2.0) ** 2
            )
        )
    )



def parse_gps_info(csv_filepath):
    """ Build a gps trajectory from a geoptis 4.3 or 4.5 csv. """
    with open(csv_filepath, "r") as f:
        reader = csv.reader(f, delimiter=";", quotechar="%")
        first_row = next(reader)
        if len(first_row) == 6:
            return parse_gps_info_v4_3(reader, first_row)
        elif len(first_row) == 10:
            return parse_gps_info_v4_5(reader, first_row)
        else:
            print("Geoptis version not recognised")


def parse_gps_info_v4_3(reader, first_row):
    """ Build a gps trajectory from a geoptis 4.3 csv. """
    trajectory = []
    video_filename = first_row[4]
    second_row = next(reader)
    geoptis_version = second_row[1]
    assert geoptis_version == "V 4.3"
    for row in reader:
        timestamp_epoch_ms = float(row[1])
        timestamp_video_ms = float(row[2])
        latitude = float(row[3])
        longitude = float(row[4])
        current = (timestamp_video_ms, longitude, latitude, timestamp_epoch_ms)
        trajectory.append(current)

    trajectory = np.array(trajectory)

    return trajectory


def parse_gps_info_v4_5(reader, first_row):
    """ Build a gps trajectory from a csv with geoptis 4.5 format.
    The original geoptis 4.5 csv needs to be split per video beforehand, 
    since the csv file used here is expected to correspond to a single video."""
    trajectory = []
    video_filename = first_row[2]
    if first_row[7].endswith("%"):
        jjson = json.loads(first_row[7][:-1])
    else:
        jjson = json.loads(first_row[7])
    geoptis_version = jjson["version"]
    assert geoptis_version == "V 4.5"
    timestamp_epoch_ms = float(first_row[3])
    timestamp_video_ms = float(first_row[8])
    longitude = float(first_row[4])
    latitude = float(first_row[5])
    current = (timestamp_video_ms, longitude, latitude, timestamp_epoch_ms)
    trajectory.append(current)
    for row in reader:
        timestamp_epoch_ms = float(row[3])
        timestamp_video_ms = float(row[8])
        longitude = float(row[4])
        latitude = float(row[5])
        current = (timestamp_video_ms, longitude, latitude, timestamp_epoch_ms)
        trajectory.append(current)

    trajectory = np.array(trajectory)

    return trajectory



def parse_videocoding(jsonpath):
    """Parse videocoding jsons."""

    def date_to_timestamp(date):
        timestamp = datetime.datetime.strptime(date, "%d%m%Y_%H%M%S").timestamp()
        return timestamp

    with open(jsonpath, "rt") as f_in:
        content = json.load(f_in)

    version = content["version"]

    classes = {r["id"]: r["name"] for r in content["rubrics"]}
    # Class -9999 is "images"
#    del classes[-9999]
    classnames = list(classes.values())
    classname_to_deg_index = {name: i for i, name in enumerate(classnames)}

    all_degradations = []
    all_timestamps = []
    for section in content["elements"][0]["sections"]:
        events = section["events"]
        # Rubric 0 is "image extraction samples", not degradation
        degradation_events = [e for e in events if e["rubric"] > 0]

        degradations = np.zeros((len(degradation_events), len(classes)))
        for i, degradation_event in enumerate(degradation_events):
            classname = classes[degradation_event["rubric"]]
            degradation_index = classname_to_deg_index[classname]
            degradations[i, degradation_index] += 1.0

        if version == "1.5":
            timestamps = [
                (e["start_timestamp"], e["end_timestamp"]) for e in degradation_events
            ]
            timestamps = np.array(timestamps, dtype=float)
            # microseconds to seconds
            timestamps *= 1e-6
        else:
            # XXX: Only version "1.4" has been tested in this case
            # XXX: We're converting dates to timestamps without timezones
            # XXX: Best precision here is seconds whereas is it in microseconds for v1.5
            timestamps = [
                (date_to_timestamp(e["start_date"]), date_to_timestamp(e["end_date"]))
                for e in degradation_events
            ]
            timestamps = np.array(timestamps, dtype=float)

        all_degradations.append(degradations)
        all_timestamps.append(timestamps)

    non_empty_count = sum([d.shape[0] != 0 for d in all_degradations])
    if non_empty_count > 0:
        all_degradations = [d for d in all_degradations if d.shape[0] != 0]
        all_timestamps = [d for d in all_timestamps if d.shape[0] != 0]
        all_degradations = np.concatenate(all_degradations, axis=0)
        all_timestamps = np.concatenate(all_timestamps, axis=0)
    else:
        all_degradations = all_degradations[0]
        all_timestamps = all_timestamps[0]

    video_start_date = content["elements"][0]["sections"][0]["start_date"]
    video_start_timestamp = date_to_timestamp(video_start_date)
    return classnames, all_degradations, all_timestamps, video_start_timestamp



def shift_timestamps_wrt_gps(geoptis_csvpath, timestamps, length_meters):
    """ Shift timestamps to timestamps from some meters before in the video."""
    trajectory = parse_gps_info(geoptis_csvpath)
    traj_times = trajectory[:, 3]
    traj_times *= 1e-3
    traj_lnglats = trajectory[:, 1:3]
    
    differentials_meters = Haversine_distance(traj_lnglats[:-1], traj_lnglats[1:])
    traveled_distances = np.cumsum(differentials_meters)
    traveled_distances = np.concatenate([[0.0], traveled_distances])
    distance_for_timestamp = interp1d(traj_times, traveled_distances)
    timestamp_for_distance = interp1d(traveled_distances, traj_times)

    # Make sure that we're in between interpolation bounds. It should be ok
    # since it's only the few first and last frames ?
    timestamps = np.clip(timestamps, a_min=traj_times[0], a_max=traj_times[-1])

    distances_at_t = distance_for_timestamp(timestamps)
    shifted_distances = distances_at_t - length_meters
    # Don't shift to before the video started
    shifted_distances[shifted_distances < 0.0] = 0.0
    shifted_timestamps = timestamp_for_distance(shifted_distances)

    return shifted_timestamps, traj_times[0]



def compare_AI_videocoding(jsonpath, videopath, geoptis_csvpath, shift_meters):
    '''
    Extract every 5 frame of the video + every frame with ponctual videocoded defect.
    Run inference on extracted images with pretrained model.
    Compute TP, FP, FN using videocoding as reference.
    '''
    classes, degradations, timestamps, start_timestamp = parse_videocoding(jsonpath)

    timestamps, traj_times_0 = shift_timestamps_wrt_gps(geoptis_csvpath, timestamps, shift_meters)
    classname_to_deg_index = {name: i for i, name in enumerate(classes)}
    deg_index_to_classname = {i: name for name, i in classname_to_deg_index.items()}

    if degradations.shape[0] == 0:
        return 0, 0

    cam = cv.VideoCapture(videopath)
    framerate = cam.get(cv.CAP_PROP_FPS)

    # load model
    if args.type == 'cls':
        model = mmcls.apis.init_model(
            args.config,
            args.checkpoint,
            device="cuda:0",
        )
    elif args.type == 'det':
        model = mmdet.apis.init_detector(
            args.config,
            args.checkpoint,
            device="cuda:0",
        )

    # TP/TN/FN array
    results = np.zeros((3, len(classes_AI)), dtype=int)

    # loop over video
    t = traj_times_0
    process_every_nth_frame = 5
    i = 0
    while True:
        i += 1
        ret, frame = cam.read()
        if not ret:
            break
        
        t += 1 / framerate

        # select every nth frame + additional frames containing single-frame defects
        count_single_in_image = 0
        videocoding_defects = np.zeros(len(classes_AI), dtype=int)
        for i_timestamp, cur_timestamps in enumerate(timestamps):
            start, end = cur_timestamps

            # continuous degradations
            if start != end and start < t < end:
                degradation_indexes = np.nonzero(degradations[i_timestamp])
                if len(degradation_indexes) != 1 and len(degradation_indexes[0]) != 1:
                    print("Several degradations in one timestamp")
                    exit(0)
                degradation_index = degradation_indexes[0][0]
                degradation_name = deg_index_to_classname[degradation_index]
                degradation_name = strip_accents(degradation_name)
                if classes_dict[degradation_name] != '':
                    idx = classes_AI[classes_dict[degradation_name]]
                    videocoding_defects[idx] = 1

            # single-frame
            if start == end and 0 < start - t < 1 / framerate:
                degradation_indexes = np.nonzero(degradations[i_timestamp])
                if len(degradation_indexes) != 1 and len(degradation_indexes[0]) != 1:
                    print("Several degradations in one timestamp")
                    exit(0)
                degradation_index = degradation_indexes[0][0]
                degradation_name = deg_index_to_classname[degradation_index]
                degradation_name = strip_accents(degradation_name)
                count_single_in_image += 1
                if classes_dict[degradation_name] != '':
                    idx = classes_AI[classes_dict[degradation_name]]
                    videocoding_defects[idx] = 1

 
        if i % process_every_nth_frame == 0 or count_single_in_image > 0:
            AI_defects = np.zeros(len(classes_AI), dtype=int)
            
            if args.type == 'cls':
                res = mmcls.apis.inference_model(model, frame, is_multi_label=True, threshold=args.threshold)            
                for pc in res["pred_class"]:
                    idx = classes_AI[pc]
                    AI_defects[idx] = 1

            elif args.type == 'det':
                res = mmdet.apis.inference_detector(model, frame)
                for ic, c in enumerate(res): # loop on classes
                    if (c[:,4] > args.threshold).sum() > 0:
                        AI_defects[ic] = 1
                
            results[0] += videocoding_defects * AI_defects == 1 # TP
            results[1] += (1-videocoding_defects) * AI_defects == 1 # FP
            results[2] += videocoding_defects * (1-AI_defects) == 1 # FN

    print(results)
    print('precision:', results[0]/(results[0]+results[1]))
    print('recall:', results[0]/(results[0]+results[2]))

    return results

    
def main(args):
    # no input video given, treat all predifined videos and combine results
    if args.inputvideo is None:
        # TP/TN/FN array
        results = np.zeros((3, len(classes_AI)), dtype=int)
        for vid, json in comp_videos.items():
            print(f'\n{vid}')
            results += compare_AI_videocoding(inputpath+'/'+json, inputpath+'/'+vid, inputpath+'/'+vid.replace('mp4', 'csv'), args.shift_meters)
        print(results)
        print('precision:', results[0]/(results[0]+results[1]))
        print('recall:', results[0]/(results[0]+results[2]))
            
    # treat a single video
    else:
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
            
        compare_AI_videocoding(args.json, args.inputvideo, args.csv, args.shift_meters)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputvideo")
    parser.add_argument("--json")
    parser.add_argument("--csv")
    parser.add_argument("--shift_meters", type=float, required=True)

    parser.add_argument("--type", choices=['cls', 'det', 'seg'], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
    
