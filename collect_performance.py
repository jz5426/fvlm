import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# === Setup paths ===
reference_csv_path = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv'
csv_directory = '/cluster/projects/mcintoshgroup/fvlm_files/rate_results'

# === Load and clean reference CSV B ===
reference_df = pd.read_csv(reference_csv_path)

test_items = [
    ['lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'],
    ['lung', 'Atelectasis', 'Not Atelectatic.', 'Atelectatic.'], 
    ['lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'],
    ['lung', 'Lung opacity', 'Not Opacity.', 'Opacity.'],
    ['lung', 'Pulmonary fibrotic sequela', 'Not Pulmonary fibrotic.', 'Pulmonary fibrotic.'],
    ['lung', 'Pleural effusion', 'Not Pleural effusion.', 'Pleural effusion.'],
    ['lung', 'Mosaic attenuation pattern', 'Not Mosaic attenuation pattern.', 'Mosaic attenuation pattern.'],
    ['lung', 'Peribronchial thickening', 'Not Peribronchial thickening.', 'Peribronchial thickening.'],
    ['lung', 'Consolidation', 'Not Consolidation.', 'Consolidation.'],
    ['lung', 'Bronchiectasis', 'Not Bronchiectasis.', 'Bronchiectasis.'],
    ['lung', 'Interlobular septal thickening', 'Not Interlobular septal thickening.', 'Interlobular septal thickening.'],
    ['heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'],
    ['heart', 'Pericardial effusion', 'Not Pericardial effusion.', 'Pericardial effusion.'],
    ['heart', 'Coronary artery wall calcification', 'Not Coronary artery wall calcification.', 'Coronary artery wall calcification.'],
    ['esophagus', 'Hiatal hernia', 'Not Hiatal hernia.', 'Hiatal hernia.'],
    ['aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.'],
]

conditions = [item[1] for item in test_items if item[1] in reference_df.columns]

# === Process each CSV file in directory ===
for file in os.listdir(csv_directory):
    if not file.endswith('.csv'):
        continue

    file_path = os.path.join(csv_directory, file)
    df = pd.read_csv(file_path)
    
    # rename the checkpoint csv column names so that it matches the column names in the reference csv file
    df = df.rename(columns={
        '_'.join(col) : col[1]
        for col in test_items 
    })

    # === Build dictionary: file_name → list of 0/1 values ===

    # get the ground truth labels follow the order from df (checkpoint csv file)
    file_to_labels = {}
    for _, row in df.iterrows():
        # get the corresponding row in the reference csv file
        file_name = row['file_name']
        ref_row = reference_df[reference_df['VolumeName'] == file_name]
        if ref_row.empty:
            assert False
        else:
            ref_row = ref_row.iloc[0]
            values = [int(ref_row[col]) for col in df.columns[1:]]
            if -1 in values:
                assert False
            file_to_labels[file_name] = values

    original_ordered_col_names = list(df.columns[1:])

    # TODO: get the predicted labels follow the orders from df. if the column value is >= 0.5 then it is 1, otherwise 0; save the results as a new dictionary like file_to_labels
    file_to_preds = {}
    for _, row in df.iterrows():
        file_name = row['file_name']
        preds = [1 if row[col] >= 0.5 else 0 for col in original_ordered_col_names]
        file_to_preds[file_name] = preds

    assert len(file_to_preds) == len(file_to_labels)

    predictedall, realall = [], []
    for key in file_to_preds:
        predict = file_to_preds[key]
        label = file_to_labels[key]
        predictedall.extend(predict)
        realall.extend(label)

    # collect the stats like ct-clip
    realall = np.rint(realall).astype(int)
    predictedall = np.rint(predictedall).astype(int)
    f1 = f1_score(realall, predictedall,average='micro')
    flat_acc = accuracy_score(realall.flatten(), predictedall.flatten())
    checkpoint_file = '_'.join(file.split('_')[-2:])
    print('{} Validation F1 Accuracy: {}; Validation Flat Accuracy: {}\n'.format(checkpoint_file, f1, flat_acc))
