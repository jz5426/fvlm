import os

def count_files_with_suffix(directory, suffix):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                # print(file)
                count += 1
    return count