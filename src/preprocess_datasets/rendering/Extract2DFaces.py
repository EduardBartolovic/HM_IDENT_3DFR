import os
from tqdm import tqdm
from pathlib import Path
import shutil


class Extract2DFaces:

    def __init__(self, root_directory, output_dir):
        self.root_directory = root_directory
        self.output_image_path = output_dir

    def extract_photos(self):
        zip_files = self.collect_obj_files()
        for i in tqdm(zip_files, desc="Gather Face Photos"):
            self.unpack_and_move(i)

    def gather_photos(self):
        files = self.collect_photos_files()
        for i in tqdm(files, desc="Copy Face Photos"):
            self.move(i)

    def unpack_and_move(self, zip_file):
        file_abspath = zip_file['zip_file_path']
        target_path = os.path.join(self.output_image_path,
                                   zip_file['user'],
                                   zip_file['date'],
                                   zip_file['scan_id'] + '_' + zip_file['scan_name'])
        os.makedirs(target_path, exist_ok=True)
        shutil.copyfile(file_abspath, os.path.join(target_path, 'tmp.zip'))
        shutil.unpack_archive(os.path.join(target_path, 'tmp.zip'), target_path)
        for root, _, files in os.walk(target_path):
            for file in files:
                if not file.endswith('.jpg'):
                    os.remove(os.path.join(target_path, file))

    def move(self, photo_file):
        file_abspath = photo_file['file_path']
        target_path = os.path.join(self.output_image_path,
                                   photo_file['user'],
                                   photo_file['date'],
                                   photo_file['scan_id'] + '_' + photo_file['scan_name'])
        os.makedirs(target_path, exist_ok=True)
        shutil.copyfile(file_abspath, os.path.join(target_path, photo_file['user']+'.png'))

    def collect_obj_files(self):
        zip_files_paths = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('photos.zip'):
                    zip_file_path = os.path.join(root, file)
                    splited_path = Path(zip_file_path).parts
                    zip_file = {'date': splited_path[-4],
                                'user': splited_path[-3],
                                'scan_id': splited_path[-2],
                                'scan_name': Path(file).stem,
                                'zip_file_path': zip_file_path}
                    zip_files_paths.append(zip_file)

        print('Collected:', len(zip_files_paths), 'Photos')
        return zip_files_paths

    def collect_photos_files(self):
        photo_files_paths = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('.png'):
                    zip_file_path = os.path.join(root, file)
                    photo_file = {'date': '1.1.1',
                                  'user': Path(file).stem,
                                  'scan_id': '0',
                                  'scan_name': '0',
                                  'file_path': zip_file_path}

                    photo_files_paths.append(photo_file)

        print('Collected:', len(photo_files_paths), 'Photos')
        return photo_files_paths
