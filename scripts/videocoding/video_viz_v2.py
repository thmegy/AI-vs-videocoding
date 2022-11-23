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

import sys


def strip_accents(s):
    # SEE: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


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
    print("first_row[7] ", first_row[7])
    if first_row[7].endswith("%"):
        jjson = json.loads(first_row[7][:-1])
    else:
        jjson = json.loads(first_row[7])
    print(" ******* ")
    print(jjson)
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
    
    differentials_meters = Harversine_distance(traj_lnglats[:-1], traj_lnglats[1:])
    traveled_distances = np.cumsum(differentials_meters)
    traveled_distances = np.concatenate([[0.0], traveled_distances])
    distance_for_timestamp = interp1d(traj_times, traveled_distances)
    timestamp_for_distance = interp1d(traveled_distances, traj_times)

    # Make sure that we're in between interpolation bounds. It should be ok
    # since it's only the few first and last frames ?
    timestamps = np.clip(timestamps, a_min=traj_times[0], a_max=traj_times[-1])

    print(traj_times[0])
    distances_at_t = distance_for_timestamp(timestamps)
    shifted_distances = distances_at_t - length_meters
    # Don't shift to before the video started
    shifted_distances[shifted_distances < 0.0] = 0.0
    shifted_timestamps = timestamp_for_distance(shifted_distances)

    return shifted_timestamps, traj_times[0]


def create_annotated_video(jsonpath, videopath, output_videopath, geoptis_csvpath, shift_meters):
    classes, degradations, timestamps, start_timestamp = parse_videocoding(jsonpath)

    print(start_timestamp)
    print(timestamps[0][0])
    timestamps, traj_times_0 = shift_timestamps_wrt_gps(geoptis_csvpath, timestamps, shift_meters)
#    sys.exit()
    classname_to_deg_index = {name: i for i, name in enumerate(classes)}
    deg_index_to_classname = {i: name for name, i in classname_to_deg_index.items()}

    is_extracting_one_class = False
    if is_extracting_one_class:
        extracted_index = None
        for name, index in classname_to_deg_index.items():
            if name == "Baches":
                extracted_index = index
                break

        degradations = degradations[:, extracted_index]
        timestamps = timestamps[:]
        timestamps = timestamps[degradations != 0].tolist()
        degradations = degradations[degradations != 0]

    if degradations.shape[0] == 0:
        return 0, 0

    cam = cv.VideoCapture(videopath)
    framerate = cam.get(cv.CAP_PROP_FPS)

    fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv.VideoWriter(
        filename=output_videopath,
        apiPreference=int(cv.CAP_FFMPEG),
        fourcc=fourcc,
        fps=int(framerate),
        frameSize=(
            int(cam.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cam.get(cv.CAP_PROP_FRAME_HEIGHT)),
        ),
        params=None,
    )

#    t = start_timestamp
    t = traj_times_0
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        t += 1 / framerate

        make_video_pause = False
        count_in_image = 0
        for i_timestamp, cur_timestamps in enumerate(timestamps):
            start, end = cur_timestamps
            # Draw continuous degradations in blue if any
            if start != end and start < t < end:
                degradation_indexes = np.nonzero(degradations[i_timestamp])
                if len(degradation_indexes) != 1 and len(degradation_indexes[0]) != 1:
                    print("Several degradations in one timestamp")
                    exit(0)
                degradation_index = degradation_indexes[0][0]
                degradation_name = deg_index_to_classname[degradation_index]
                degradation_name = strip_accents(degradation_name)
                cv.putText(
                    frame,
                    degradation_name,
                    (frame.shape[1] - 1235, frame.shape[0] - 70 * (count_in_image + 1)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 50, 50),
                    2,
                    cv.LINE_AA,
                )
                count_in_image += 1
                count += 1

            # Draw single-frame degradations in red if any
            if start == end and 0 < start - t < 1 / framerate:
                degradation_indexes = np.nonzero(degradations[i_timestamp])
                if len(degradation_indexes) != 1 and len(degradation_indexes[0]) != 1:
                    print("Several degradations in one timestamp")
                    exit(0)
                degradation_index = degradation_indexes[0][0]
                degradation_name = deg_index_to_classname[degradation_index]
                degradation_name = strip_accents(degradation_name)
                cv.putText(
                    frame,
                    degradation_name,
                    (frame.shape[1] - 1235, frame.shape[0] - 70 * (count_in_image + 1)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2,
                    (50, 50, 255),
                    2,
                    cv.LINE_AA,
                )
                count_in_image += 1
                count += 1
                make_video_pause = True

        # Pause image for 1s if needed
        if make_video_pause:
            for i in range(30):
                writer.write(frame)
        else:
            writer.write(frame)

    writer.release()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputvideo", required=True)
    parser.add_argument("--outputvideo", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--shift_meters", type=float, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.csv != args.inputvideo.replace('mp4', 'csv'):
        csv_path = args.csv.split('/')
        csv_path = '/'.join(csv_path[:-1])
        df = pd.read_csv(args.csv, delimiter=';', header=None, quotechar='%')
        for vid in pd.unique(df[0]):
            df[df[0]==vid].to_csv(f'{csv_path}/{vid}.csv', header=False, sep=';', index=False, quotechar='%')
    
    create_annotated_video(args.json, args.inputvideo, args.outputvideo, args.inputvideo.replace('mp4', 'csv'), args.shift_meters)
