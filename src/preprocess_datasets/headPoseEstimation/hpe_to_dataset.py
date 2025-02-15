import csv
import hashlib
import os
import shutil

import numpy as np


def read_file(file_path, remove_header=True):
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        if remove_header:
            header = next(reader)
        for row in reader:
            data.append(row)

    #print("Header:", header)
    #print("Data:")
    #for row in data:
    #    print(row)

    return np.array(data)


def generate_voxceleb_dataset(folder_root, input_folder, dataset_output_folder):
    counter = 0
    for root, _, files in os.walk(os.path.join(folder_root)):
        if "hpe" in root:
            for txt_file in files:
                if txt_file.endswith('matched_angles.txt'):
                    file_path = os.path.join(root, txt_file)
                    data = read_file(file_path)

                    folder_image_path = os.path.join(root, "..", input_folder)

                    sample_name = os.path.abspath(os.path.join(root, os.pardir))
                    id_name = os.path.abspath(os.path.join(sample_name, os.pardir))
                    sample_name = os.path.basename(sample_name)
                    id_name = os.path.basename(id_name)

                    destination = os.path.join(dataset_output_folder, id_name)#, sample_name)
                    os.makedirs(destination, exist_ok=True)

                    for info in data:
                        src = os.path.join(folder_image_path, info[6])
                        hash_name = hashlib.sha1((id_name + sample_name).encode()).hexdigest()
                        #dst = os.path.join(destination, f'{hash_name}{info[0]}_{info[1]}_{info[7]}_image.jpg') # With Error
                        dst = os.path.join(destination, f'{hash_name}{info[0]}_{info[1]}_image.jpg')
                        shutil.copy(src, dst)
                        counter += 1
    print("Copied", counter, "files")

if __name__ == '__main__':
    input_folder = "E:\\Download\\face\\VoxCeleb1_test"  # Folder containing original preprocessed files
    dataset_output_folder = "E:\\Download\\face\\VoxCeleb1_test_dataset"

    generate_voxceleb_dataset(input_folder, "face_cropped", dataset_output_folder)
