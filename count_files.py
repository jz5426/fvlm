import os

def count_files_with_suffix(directory, suffix):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                count += 1
    return count

image_root = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_train_fix/'
mask_root = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_train_mask/'
file_suffix = ".nii.gz"  # or ".txt", "_mask.nii.gz", etc.
num_files = count_files_with_suffix(image_root, file_suffix)
print(f"Number of files ending with '{file_suffix}': {num_files}") # make sense
num_files = count_files_with_suffix(mask_root, file_suffix)
print(f"Number of files ending with '{file_suffix}': {num_files}") # make sense

# # Example usage:
# directory_path = "/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_val"
# file_suffix = ".h5"  # or ".txt", "_mask.nii.gz", etc.
# num_files = count_files_with_suffix(directory_path, file_suffix)
# print(f"Number of files ending with '{file_suffix}': {num_files} [ORIGINAL]") # make sense

# directory_path = "/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_val_fixed"
# file_suffix = ".nii.gz"  # or ".txt", "_mask.nii.gz", etc.
# num_files = count_files_with_suffix(directory_path, file_suffix)
# print(f"Number of files ending with '{file_suffix}': {num_files} [FIXED]") # make sense

# directory_path = "/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_val_mask"
# file_suffix = ".nii.gz"  # or ".txt", "_mask.nii.gz", etc.
# num_files = count_files_with_suffix(directory_path, file_suffix)
# print(f"Number of files ending with '{file_suffix}': {num_files} [MASK]") # make sense

# directory_path = "/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_merged_val_masks"
# file_suffix = ".nii.gz"  # or ".txt", "_mask.nii.gz", etc.
# num_files = count_files_with_suffix(directory_path, file_suffix)
# print(f"Number of files ending with '{file_suffix}': {num_files} [MERGED MASK]")

# directory_path = "/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_resized_val_images"
# file_suffix = ".nii.gz"  # or ".txt", "_mask.nii.gz", etc.
# num_files = count_files_with_suffix(directory_path, file_suffix)
# print(f"Number of files ending with '{file_suffix}': {num_files} [RESIZED TRAIN IMAGE]")

# directory_path = "/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_resized_val_masks"
# file_suffix = ".nii.gz"  # or ".txt", "_mask.nii.gz", etc.
# num_files = count_files_with_suffix(directory_path, file_suffix)
# print(f"Number of files ending with '{file_suffix}': {num_files} [RESIZED TRAIN MASK]")

# directory_path = "/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_processed_val_images"
# file_suffix = ".nii.gz"  # or ".txt", "_mask.nii.gz", etc.
# num_files = count_files_with_suffix(directory_path, file_suffix)
# print(f"Number of files ending with '{file_suffix}': {num_files} [PROCESSED TRAIN IMAGE]")

# directory_path = "/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_processed_val_masks"
# file_suffix = ".nii.gz"  # or ".txt", "_mask.nii.gz", etc.
# num_files = count_files_with_suffix(directory_path, file_suffix)
# print(f"Number of files ending with '{file_suffix}': {num_files} [PROCESSED TRAIN MASK]")