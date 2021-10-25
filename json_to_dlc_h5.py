import pandas as pd
import json
import csv
import numpy as np




json_annotation_path = "/media/storage2/open_monkey/monkey_train_annotations.json"
csv_annotation_path = "monkey_train_annotations.csv"
h5_annotation_path = "/media/storage2/open_monkey/monkey_train_annotations.h5"


with open(json_annotation_path, "r") as fh:
    annotations = json.load(fh)["data"]


with open(csv_annotation_path, 'w', newline='\n') as csvfile:
    monkeywriter = csv.writer(csvfile)
    monkeywriter.writerow(['scorer'] + ['n/a'] * 17)
    bodyparts = ['right_eye', 'left_eye', 'nose', 'head', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist', 'hip', 'right_knee', 'right_ankle', 'left_knee', 'left_ankle', 'tail']
    row_body_parts = ['body_parts']
    for part in bodyparts:
        row_body_parts += [part]
        row_body_parts += [part]
    print(row_body_parts)
    monkeywriter.writerow(row_body_parts)

    coord_row = ["coords"] + ["x", "y"] * 17

    monkeywriter.writerow(coord_row)

    for annotation in annotations:
        row = []
        row.append(annotation["file"])
        keypoints = [float(point) for point in annotation["landmarks"]]
        row += keypoints
        monkeywriter.writerow(row)

annotations = pd.read_csv(csv_annotation_path, dtype=str)


annotations.loc[:, 'scorer':].to_hdf(h5_annotation_path, 'data', mode='w', format="table")
