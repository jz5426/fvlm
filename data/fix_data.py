import ast
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tqdm
import h5py

def process_row(row):
    # Set up directory parameters
    VolumeName = row["VolumeName"]
    dir1 = VolumeName.rsplit("_", 1)[0]
    dir2 = VolumeName.rsplit("_", 2)[0]
    #TODO: organize the files in the following fomrat if necessary.
    # filepath = os.path.join(data_root, f"{split}", dir2, dir1, VolumeName)

    # transform from "train_1_a" to "train_1a" NOTE:TODO: this is just temporary changes, the file structure should follows exactly the same as the one from metadata file ideally.
    dir1 = dir1[::-1].replace("_", "", 1)[::-1] 
    # Handle compound extensions like .nii.gz
    base, ext = os.path.splitext(VolumeName)
    if ext == ".gz":
        base, _ = os.path.splitext(base)  # strip .nii as well
    VolumeName = base + ".h5"
    filepath = os.path.join(f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_{split}', dir2, dir1, VolumeName)
    dirpath = os.path.dirname(filepath)
    dirpath = dirpath.replace(f"/CTRATE_Volumes_raw_h5_fp16_noflip_{split}/", f"/CTRATE_Volumes_raw_h5_fp16_noflip_{split}_fixed/")

    # skip the files if not exists in the data directory
    if not os.path.exists(filepath):
        return 

    if os.path.exists(os.path.join(dirpath, os.path.basename(filepath))):
        return

    # Read Image
    if filepath.endswith('h5'):
        with h5py.File(filepath, "r") as f:
            image_np = f["ct"][:].astype(np.float32) # a numpy array (512, 512, 303), which is the original shape, by convention it is x,y,z
            image_np = np.transpose(image_np, (2, 1, 0)) # become z, y, x
        image = sitk.GetImageFromArray(image_np)  # NOTE: This assumes axis order is z, y, x but image is in axis order of x, y, z

    # image = sitk.ReadImage(filepath) # assume axis order of x, y, z

    # Set Spacing
    (x, y), z = map(float, ast.literal_eval(row["XYSpacing"])), row["ZSpacing"]
    image.SetSpacing((x, y, z))

    # Set Origin
    image.SetOrigin(ast.literal_eval(row["ImagePositionPatient"]))

    # Set Direction
    orientation = ast.literal_eval(row["ImageOrientationPatient"])
    row_cosine, col_cosine = orientation[:3], orientation[3:6]
    z_cosine = np.cross(row_cosine, col_cosine).tolist()
    image.SetDirection(row_cosine + col_cosine + z_cosine)

    # Fix Rescale
    RescaleIntercept = row["RescaleIntercept"]
    RescaleSlope = row["RescaleSlope"]
    adjusted_hu = image * RescaleSlope + RescaleIntercept

    # Convert the image to int16
    adjusted_hu = sitk.Cast(adjusted_hu, sitk.sitkInt16)

    # Write Image
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    # replace
    # NOTE: totalsegmentor requires nifitimage file format to do the the segmentation
    base, ext = os.path.splitext(filepath)
    base += '.nii.gz'
    # np.transpose(np_image_transposed, (1, 2, 0)) # reverse the previous transpose operation
    sitk.WriteImage(adjusted_hu, os.path.join(dirpath, os.path.basename(base)))


if __name__ == "__main__":

    # /cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/original/train_1_a_1.nii.gz
    # img = sitk.ReadImage('/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/original/train_1_a_1.nii.gz') # in shape (512, 512, 303)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--split", required=False, default='train', type='str')
    # args = parser.parse_args()
    # split = args.split

    split = 'train'
    d = "validation" if split == "valid" else "train"
    
    data_root = Path("/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/")
    metadata = pd.read_csv(os.path.join(data_root, f"metadata/{d}_metadata.csv"))
    rows = [row[1] for row in metadata.iterrows()]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(process_row, rows), total=len(rows)))
    # process_row(rows[0])
    print('finished')
