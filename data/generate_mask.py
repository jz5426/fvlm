import os
import sys
from pathlib import Path
import nibabel as nib
import pandas as pd
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.TotalSegmentator.totalsegmentator.python_api import totalsegmentator

def generate_mask(row):
    VolumeName = row["VolumeName"]
    dir1 = VolumeName.rsplit("_", 1)[0]
    dir2 = VolumeName.rsplit("_", 2)[0]

    # filepath = os.path.join(data_root, f"{split}_fixed", dir2, dir1, VolumeName)
    # dirpath = os.path.dirname(filepath)
    # dirpath = dirpath.replace(f"/{split}_fixed/", f"/{split}_mask/")

    # transform from "train_1_a" to "train_1a" NOTE:TODO: this is just temporary changes, the file structure should follows exactly the same as the one from metadata file ideally.
    dir1 = dir1[::-1].replace("_", "", 1)[::-1] 
    filepath = os.path.join(f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_{split}_fixed', dir2, dir1, VolumeName)
    dirpath = os.path.dirname(filepath)
    dirpath = dirpath.replace(f"/CTRATE_Volumes_raw_h5_fp16_noflip_{split}_fixed/", f"/CTRATE_Volumes_raw_h5_fp16_noflip_{split}_mask/")

    # skip the files if not exists in the data directory
    if not os.path.exists(filepath):
        return 
    # check if the file is preprocessed already
    if os.path.exists(os.path.join(dirpath, os.path.basename(filepath))):
        return

    # NOTE: extension should be .nii.gz, not the h5 version.
    input_img = nib.load(filepath)
    output_img = totalsegmentator(input_img, quiet=True)

    Path(dirpath).mkdir(parents=True, exist_ok=True)
    nib.save(output_img, os.path.join(dirpath, os.path.basename(filepath)))
    print(f'finish processing {os.path.join(dirpath, os.path.basename(filepath))}')


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Training")
    # parser.add_argument("--split", required=False, default='train', type='str')
    # args = parser.parse_args()
    # split = args.split

    #NOTE: depends on the _fix data

    split = 'val'
    d = "validation" if split == "val" else "train"
    data_root = Path("/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/")
    # metadata = pd.read_csv(os.path.join(data_root, f"metadata/{d}_metadata.csv")) # TODO: uncomment this when the val splits actually comes from the val_metadata
    metadata = pd.read_csv(os.path.join(data_root, f"metadata/train_metadata.csv"))

    rows = [row[1] for row in metadata.iterrows()]
    for row in rows:
        generate_mask(row)
    # generate_mask(rows[0])
    print('finished generate_mask.py script')

