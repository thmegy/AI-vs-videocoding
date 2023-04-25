import numpy as np
import unicodedata
import csv
import datetime
import json
from scipy.interpolate import interp1d
import cv2 as cv
import os
import glob
import sys
import tqdm


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

    classnames = [] # including subclasses (e.g. gravité)
    classes = {}
    for r in content["rubrics"]:
        if r["id"]==-9999:
            continue
        if r['name'][-2] == ' ': # remove initial of videocoder if present
            r['name'] = r['name'][:-2]

        classes[r["id"]] = r["name"]
            
        for sub in r['parts'][0]['lexicons']:
            classnames.append(f'{r["name"]} {sub["value"]}')
    classname_to_deg_index = {name: i for i, name in enumerate(classnames)}

    all_degradations = []
    all_timestamps = []
    for section in content["elements"][0]["sections"]:
        events = section["events"]
        degradation_events = [e for e in events if e["rubric"] >= 0]
        
        degradations = np.zeros((len(degradation_events), len(classnames)))
        for i, degradation_event in enumerate(degradation_events):
            class_id = degradation_event['rubric']
            sub_class = degradation_event['parts'][str(class_id)]
            classname = f'{classes[class_id]} {sub_class}'
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

    return classnames, all_degradations, all_timestamps



def get_length_timestamp_map(geoptis_csvpath):
    """Get relation between timestamp and length travelled."""
    trajectory = parse_gps_info(geoptis_csvpath)
    traj_times = trajectory[:, 3]
    traj_times *= 1e-3
    traj_lnglats = trajectory[:, 1:3]
    
    differentials_meters = Haversine_distance(traj_lnglats[:-1], traj_lnglats[1:])
    traveled_distances = np.cumsum(differentials_meters)
    traveled_distances = np.concatenate([[0.0], traveled_distances])
    distance_for_timestamp = interp1d(traj_times, traveled_distances)

    return traj_times[0], distance_for_timestamp, traveled_distances[-1]



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



def compute_distances(length_vid, length_AI, AI_score, N_ai):
    # compute distances for each element of length_vid wrt length_AI (arrays or lists)
    # For each degradation in length_vid measure and return the N_ai smallest distances wrt to AI detections
    # returns array of distances (N_degradation_vid, N_ai), and array of AI scores (same dimensions)
    assert len(length_AI) == len(AI_score)
    
    distance_list = []
    score_list = []
    for l1 in length_vid:
        if len(length_AI) > 0:
            dist = np.abs(l1 - np.array(length_AI))
            if len(length_AI) >= N_ai:
                sorted_inds = np.argsort(dist) # sort by distance
                dist = dist[sorted_inds]
                dist = dist[:N_ai]

                score = AI_score[sorted_inds]
                score = score[:N_ai]
            else:
                dist_tmp = 50 * np.ones(N_ai)
                dist_tmp[:len(dist)] = dist

                score = np.zeros(N_ai)
                score[:len(dist)] = AI_score
                
                dist = dist_tmp
        else:
            dist = 50 * np.ones(N_ai)
            score = np.zeros(N_ai)

        distance_list.append(dist)
        score_list.append(score)
    return np.stack(distance_list), np.stack(score_list)





def extract_lengths_videocoding(jsonpath, geoptis_csvpath, classes_vid,
                                classes_comp, process_every_nth_meter, return_max_distance=False):
    '''
    Return length of videocoded degradations from start of mission.
    '''
    classes, degradations, timestamps = parse_videocoding(jsonpath)

    miss_count = 0
    for cls in classes:
        if strip_accents(cls) not in classes_vid.keys():
            miss_count += 1
            print(strip_accents(cls))
    if miss_count > 0:
        print('\nVideocoded classes do not correspond to classes given with --cls_config...')
        print('Expected classes are:')
        for cls in classes:
            print(strip_accents(cls))
        print('')
        sys.exit()

    traj_times_0, distance_for_timestamp, max_distance = get_length_timestamp_map(geoptis_csvpath)
    classname_to_deg_index = {name: i for i, name in enumerate(classes)}
    deg_index_to_classname = {i: name for name, i in classname_to_deg_index.items()}

    # length list for each AI class
    length_video = [[] for _ in range(len(classes_comp))]

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

    if return_max_distance:
        return length_video, max_distance
    else:
        return length_video




def extract_lengths_AI(videopath, geoptis_csvpath, classes_AI, classes_comp, ai_type, ai_config,
                      ai_checkpoint, process_every_nth_meter, device='cuda:0'):
    '''
    Extract a frame of the video every n meter.
    Run inference on extracted images with pretrained model.
    Return length of detected degradations from start of mission.
    '''
    vid_name = videopath.split('/')[-1].replace(".mp4","")
    extract_path = f'prepare_annotations/processed_videos/{vid_name}/{process_every_nth_meter}'
    outfilename = f'{extract_path}/releve_IA.json'
    if not os.path.isfile(outfilename):

        traj_times_0, distance_for_timestamp, max_distance = get_length_timestamp_map(geoptis_csvpath)

        # length list for each AI class
        length_AI = [[] for _ in range(len(classes_comp))]
        length_AI_score = [[] for _ in range(len(classes_comp))] # score associated to prediction

        cam = cv.VideoCapture(videopath)

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
                cv.imwrite(f'{extract_path}/{d}.jpg', frame)


        extracted_frames = glob.glob(f'{extract_path}/*.jpg')

        # load model
        if ai_type == 'cls':
            import mmcls.apis
            model = mmcls.apis.init_model(
                ai_config,
                ai_checkpoint,
                device=device,
            )
        elif ai_type == 'det':
            import mmdet.apis
            model = mmdet.apis.init_detector(
                ai_config,
                ai_checkpoint,
                device=device,
            )
        elif ai_type == 'seg':
            import mmseg.apis
            model = mmseg.apis.init_segmentor(
                ai_config,
                ai_checkpoint,
                device=device,
            )

        # run inference on extracted frames
        for fname in tqdm.tqdm(extracted_frames):
            d = float(fname.split('/')[-1].replace('.jpg', ''))
            frame = cv.imread(fname)

            if ai_type == 'cls':
                res = mmcls.apis.inference_model(model, frame, is_multi_label=True, threshold=0.01)
                for pc, ps in zip(res['pred_class'], res['pred_score']):
                    if classes_AI[pc] != '':
                        idx = classes_comp[classes_AI[pc]]
                        length_AI[idx].append(d)
                        length_AI_score[idx].append(ps)

            elif ai_type == 'det':
                ann = []

                try:
                    res = mmdet.apis.inference_detector(model, frame)
                except:
                    print(fname)
                    os.system(f'rm "{fname}"')
                    continue
                image_width = frame.shape[1]
                image_height = frame.shape[0]
                for ic, c in enumerate(res): # loop on classes
                    if (c[:,4] > 0.01).sum() > 0:

                        degradation_name = list(classes_AI.items())[ic][1]
                        if degradation_name != '':
                            idx = classes_comp[degradation_name]

                            length_AI[idx].append(d)
                            length_AI_score[idx].append(c[:,4].max().item()) # take highest score if several instances of same class

        outdict = {}
        outdict['length'] = length_AI
        outdict['score'] = length_AI_score
        with open(outfilename, 'w') as f:
            json.dump(outdict, f)
    else:
        with open(outfilename, 'r') as f:
            indict = json.load(f)
        length_AI = indict['length']
        length_AI_score = indict['score']

    return length_AI, length_AI_score, extract_path
