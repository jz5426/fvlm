import torch
from monai import transforms
from pathlib import Path
import numpy as np
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import shutil

from count_files import count_files_with_suffix

def process_image(loader, mask_path):
    # mask_path is resized_{phrase}_images
    original_mask_path = mask_path
    img_path = None
    try:
        phase = 'train' if 'train_mask' in mask_path else 'val'

        # mask_path = mask_path.replace(
        #     f"{phase}_mask", 
        #     f"resized_{phase}_masks")
        img_path = mask_path.replace("masks", "images") # resized_{phrase}_images

        # if (
        #     Path(
        #         img_path.replace(
        #             f"resized_{phase}_images", 
        #             f"processed_{phase}_images")
        #     ).exists()
        #     and Path(
        #         mask_path.replace(
        #             f"resized_{phase}_masks", 
        #             f"processed_{phase}_masks")
        #     ).exists()
        # ): 
        #     # print('Skipping ', img_path)
        #     return None
        
        trans_input = {"image": img_path, "label": mask_path}
        # load = transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True)
        # transpose = transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1))
        # scaleIntensity = transforms.ScaleIntensityRanged(
        #     keys=["image"], a_min=-1150, a_max=350,
        #     b_min=0.0, b_max=1.0, clip=True
        # )
        data = loader(trans_input)

        image = data["image"]
        label = data["label"]
        
        old_unique_organ_ids = label.unique()

        roi_coords = np.nonzero(label[0])
        min_dhw = torch.from_numpy(np.min(roi_coords, axis=1))
        max_dhw = torch.from_numpy(np.max(roi_coords, axis=1))

        extend_d = 5
        extend_hw = 20

        min_dhw = torch.maximum(
            min_dhw - torch.tensor([extend_d, extend_hw, extend_hw]),
            torch.tensor([0, 0, 0]),
        )

        max_dhw = torch.minimum(
            max_dhw + torch.tensor([extend_d, extend_hw, extend_hw]),
            torch.tensor([image.shape[1], image.shape[2], image.shape[3]]),
        )

        data["image"] = image[
            :, min_dhw[0] : max_dhw[0], min_dhw[1] : max_dhw[1], min_dhw[2] : max_dhw[2]
        ]
        data["label"] = label[
            :, min_dhw[0] : max_dhw[0], min_dhw[1] : max_dhw[1], min_dhw[2] : max_dhw[2]
        ]

        new_unique_organ_ids = data["label"].unique()

        assert torch.all(old_unique_organ_ids == new_unique_organ_ids)

        saver = transforms.Compose(
            [
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=(112, 256, 352),
                    mode="constant",
                    constant_values=0
                ),
                transforms.SpatialPadd(
                    keys=["label"],
                    spatial_size=(112, 256, 352),
                    mode="constant",
                    constant_values=0
                ),
                transforms.SaveImaged(
                    output_dir=str(
                        Path(
                            img_path.replace(
                                f"resized_{phase}_images", f"processed_{phase}_images")
                        ).parent
                    ),
                    keys=["image"],
                    output_postfix="",
                    separate_folder=False,
                    resample=False,
                    dtype=np.float16 # TODO: make sure that with sangwook of the right datatype for space shrinking
                ),
                transforms.SaveImaged(
                    output_dir=str(
                        Path(
                            mask_path.replace(
                                f"resized_{phase}_masks", f"processed_{phase}_masks")
                        ).parent
                    ),
                    keys=["label"],
                    output_postfix="",
                    separate_folder=False,
                    resample=False,
                ),
            ]
        )
        saver(data)
    except Exception as e:
        print('Error', e, img_path)
        return

    # remove the mask from the resized_mask folder if successfully preprocessed.
    if os.path.isfile(original_mask_path):
        os.remove(original_mask_path)
        print(f"Removed image mask file {original_mask_path} from resized_mask_path folder")
    else:
        print(f"image mask file {original_mask_path} does not exist.")


    # remove the image from the resized_mask folder if successfully preprocessed.
    if os.path.isfile(img_path):
        os.remove(img_path)
        print(f"Removed image file {img_path} from resized_image_path folder")
    else:
        print(f"image file {img_path} does not exist.")

# the following replace the original patient_paths implementation above
def _find_second_level_dirs(root_dir, substring=None):
    second_level_dirs = []

    for first_level in os.listdir(root_dir):
        first_path = os.path.join(root_dir, first_level)
        if os.path.isdir(first_path):
            for second_level in os.listdir(first_path):
                second_path = os.path.join(first_path, second_level)
                if os.path.isdir(second_path):
                    if substring is None or substring in second_level:
                        second_level_dirs.append(second_path)

    return second_level_dirs

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--split", required=False, default='train', type='str')
    # args = parser.parse_args()
    # split = args.split
    # image_root = f"{split}_fix"
    # mask_root = f"{split}_mask"

    # NOTE: depends on the resized_{phrase}_images and resized_{phrase}_masks data

    split = 'train'
    mask_root = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_resized_{split}_masks/'    
    image_root = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_resized_{split}_images/'
    # patient_paths = _find_second_level_dirs(image_root, 'train_')
    # np.save("/cluster/projects/mcintoshgroup/fvlm_files/decomposed_report/patient_paths.npy", np.array(patient_paths))

    mask_paths = []
    for root, _, files in os.walk(mask_root):
        for file in files:
            mask_paths.append(os.path.join(root, file))

    loader = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            )
        ])

    max_workers = 8
    func = partial(process_image, loader)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(func, mask_paths), total=len(mask_paths)):
            pass

    process_image(loader, mask_paths[0])

    # remove the unnecessary directories
    if os.path.isdir(image_root) and count_files_with_suffix(image_root, '.nii.gz') == 0:
        shutil.rmtree(image_root)
        print(f"{image_root} removed.")

    if os.path.isdir(mask_root) and count_files_with_suffix(mask_root, '.nii.gz') == 0:
        shutil.rmtree(mask_root)
        print(f"{mask_root} removed.")
    
    print('finished preprocess.py script')
    