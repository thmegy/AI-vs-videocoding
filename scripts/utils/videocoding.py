import numpy as np
import unicodedata
import csv
import datetime
import json
from scipy.interpolate import interp1d


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
    classnames = [] # including subclasses (e.g. gravité)
    for r in content["rubrics"]:
        for sub in r['parts'][0]['lexicons']:
            classnames.append(f'{r["name"]} {sub["value"]}')
    classname_to_deg_index = {name: i for i, name in enumerate(classnames)}

    all_degradations = []
    all_timestamps = []
    for section in content["elements"][0]["sections"]:
        events = section["events"]
        # Rubric 0 is "image extraction samples", not degradation
        degradation_events = [e for e in events if e["rubric"] > 0]

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

    return traj_times[0], distance_for_timestamp
