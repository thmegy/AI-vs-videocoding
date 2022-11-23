import argparse
import datetime
import json
import os
import re
import csv

import cv2 as cv
import numpy as np
import tqdm
from scipy.interpolate import interp1d


def Harversine_distance(lnglats1, lnglats2):
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


def parse_gps_info_v4_3(csv_filepath):
    """ Build a gps trajectory from a geoptis 4.3 csv. """
    trajectory = []
    with open(csv_filepath, "r") as f:
        reader = csv.reader(f, delimiter=";", quotechar="%")
        first_row = next(reader)
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


def get_timestamps_wrt_gps(geoptis_csvpath, timestamps):
    """ Get distance as a function of time and reverse, thanks to gps coordinates."""
    trajectory = parse_gps_info_v4_3(geoptis_csvpath)
    traj_times = trajectory[:, 3]
    traj_times *= 1e-3
    traj_lnglats = trajectory[:, 1:3]

    differentials_meters = Harversine_distance(traj_lnglats[:-1], traj_lnglats[1:])
    traveled_distances = np.cumsum(differentials_meters)
    traveled_distances = np.concatenate([[0.0], traveled_distances])
    distance_for_timestamp = interp1d(traj_times, traveled_distances)
    timestamp_for_distance = interp1d(traveled_distances, traj_times)

    return timestamp_for_distance, distance_for_timestamp, traj_times[0], traj_times[-1], traveled_distances[-1]


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
    del classes[-9999]
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

    return classnames, all_degradations, all_timestamps


def extract_degradation_images(args, jsonpath, videopath, csvpath, outputpath):
    os.makedirs(outputpath, exist_ok=True)
    
    classes, degradations, timestamps = parse_videocoding(jsonpath)

    timestamp_for_distance, distance_for_timestamp, start_timestamp, max_time, max_distance = get_timestamps_wrt_gps(csvpath, timestamps)
    classname_to_deg_index = {name: i for i, name in enumerate(classes)}
    deg_index_to_classname = {i: name for name, i in classname_to_deg_index.items()}

    is_extracting_one_class = False
    if is_extracting_one_class:
        extracted_index = None
        for name, index in classname_to_deg_index.items():
            if name  == "Baches":
                extracted_index = index
                break

        degradations = degradations[:, extracted_index]
        timestamps = timestamps[:]
        timestamps = timestamps[degradations != 0].tolist()
        degradations = degradations[degradations != 0]

    if degradations.shape[0] == 0:
        return 0, 0

    cam = cv.VideoCapture(videopath)
    t = start_timestamp
    framerate = cam.get(cv.CAP_PROP_FPS)

    process_every_nth_frame = 10
    i = 0
    count = 0

    label_dict = {}
    shift_frames_plus = {}
    shift_frames_minus = {}
                
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        t += 1 / framerate

        i += 1

        export_fname = os.path.basename(videopath).replace(".mp4", f"-{i:06}.jpg").replace(' ', '_')
        label_dict[export_fname] = []
        label_fname = os.path.basename(videopath).replace(".mp4", f".json").replace(' ', '_')
    
        for i_timestamp, cur_timestamps in enumerate(timestamps):            
            start, end = cur_timestamps
            
            if (start != end and start < t < end) or (start == end and 0 < start - t < 1 / framerate):                
                degradation_indexes = np.nonzero(degradations[i_timestamp])
                if len(degradation_indexes) != 1 and len(degradation_indexes[0]) != 1:
                    print("YOYOYO pas qu'une dégradation dans le timestamp")
                    exit(0)
                degradation_index = degradation_indexes[0][0]
                degradation_name = deg_index_to_classname[degradation_index]
                label_dict[export_fname].append(degradation_name)
                
                if start != end and i % process_every_nth_frame != 0:
                    label_dict[export_fname].remove(degradation_name)
                    continue
                
                export_fpath = os.path.join(outputpath, export_fname)
                count += 1
                cv.imwrite(export_fpath, frame)

                if args.do_shift:
                    if args.shift_type=='distance':
                        distance_at_t = distance_for_timestamp(t)
                        shifted_distance_minus = distance_at_t - args.shift
                        shifted_distance_plus = distance_at_t + args.shift
                        # Don't shift to before the video started
                        if shifted_distance_minus < 0:
                            shifted_distance_minus = 0.
                        # Don't shift to after the video ended
                        if shifted_distance_plus > max_distance:
                            shifted_distance_plus = max_distance
                
                        shift_frames_minus[float(timestamp_for_distance(shifted_distance_minus))] = export_fpath
                        shift_frames_plus[float(timestamp_for_distance(shifted_distance_plus))] = export_fpath

                    elif args.shift_type=='time':
                        shifted_time_minus = t - args.shift
                        shifted_time_plus = t + args.shift
                        # Don't shift to before the video started
                        if shifted_time_minus < 0:
                            shifted_time_minus = 0.
                        # Don't shift to after the video ended
                        if shifted_time_plus > max_time:
                            shifted_time_plus = max_time
                
                        shift_frames_minus[shifted_time_minus] = export_fpath
                        shift_frames_plus[shifted_time_plus] = export_fpath
                
        if len(label_dict[export_fname]) == 0:
            if np.random.binomial(1, 0.001, 1) == 1:
                label_dict[export_fname].append('Background')
                export_fpath = os.path.join(outputpath, export_fname)
                cv.imwrite(export_fpath, frame)                
            else:
                del label_dict[export_fname]
                
    with open(os.path.join(outputpath, label_fname), 'w') as f:
        json.dump(label_dict, f)


    # loop second time of frames to export images shifter by +- <shift_meters>
    if args.do_shift:
        cam.set(1, -1)
        t = start_timestamp
        i = 0
    
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            t += 1 / framerate

            for tplus, path in shift_frames_plus.items():
                if abs(tplus - t) < 1 / framerate:
                    cv.imwrite(path.replace('.jpg', f'_plus.jpg'), frame)

            for tminus, path in shift_frames_minus.items():
                if abs(tminus - t) < 1 / framerate:
                    cv.imwrite(path.replace('.jpg', f'_minus.jpg'), frame)

    return count, degradations.shape[0]



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videodir", required=True)
    parser.add_argument("--jsondir", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--do-shift", action='store_true', help='Extract images shifted by time or distance for each annotation, in addition to original images')
    parser.add_argument("--shift-type", type=str, choices=['time', 'distance'], help='Shift by time or distance')
    parser.add_argument("--shift", type=float, default=0., help='time shift in seconds, or distance shift in meters')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Find jsons
    jsonpaths = []
    for root, dirnames, filenames in os.walk(args.jsondir):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext != ".json":
                continue
            jsonpath = os.path.join(root, filename)
            jsonpaths.append(jsonpath)

    # Find videos
    videopaths = []
    for root, dirnames, filenames in os.walk(args.videodir):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext != ".mp4":
                continue
            videopath = os.path.join(root, filename)
            videopaths.append(videopath)

    # Order jsons using their beginning time as key
    pattern = re.compile(".*_(\d{8}_\d{6}_\d{3})\.json")
    jsonpath_for_timeid = {}
    for jsonpath in jsonpaths:
        timeids = pattern.findall(jsonpath)
        if len(timeids) != 1:
            raise RuntimeError
        timeid = timeids[0]
        if timeid in jsonpath_for_timeid:
            # If there are two json files for a video, pick the last one edited
            selected = sorted([jsonpath_for_timeid[timeid], jsonpath])[1]
            jsonpath_for_timeid[timeid] = selected
            continue
        jsonpath_for_timeid[timeid] = jsonpath


    # Order videos using their beginning time as key
    pattern = re.compile(".*_(\d{8}_\d{6}_\d{3})\.mp4")
    videopath_for_timeid = {}
    for videopath in videopaths:
        timeids = pattern.findall(videopath)
        if len(timeids) != 1:
            raise RuntimeError
        timeid = timeids[0]
        if timeid in videopath_for_timeid:
            raise RuntimeError
        videopath_for_timeid[timeid] = videopath

    # Run image extraction on all couples
    count = 0
    total = 0
    for timeid, jsonpath in tqdm.tqdm(jsonpath_for_timeid.items()):
        if timeid not in videopath_for_timeid:
            raise RuntimeError
        videopath = videopath_for_timeid[timeid]
        csvpath = videopath.replace('.mp4', '.csv')
        cur_count, cur_total = extract_degradation_images(args, jsonpath, videopath, csvpath, args.outputdir)
        count += cur_count
        total += cur_total
    print(f"Exported {count} / {total} degradations")


