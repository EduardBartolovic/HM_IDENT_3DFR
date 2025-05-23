import time

import os
from tqdm import tqdm
from pathlib import Path

from src.preprocess_datasets.rendering.Facerender import render


class ObjFileRenderer:

    def __init__(self, root_directory, output_dir, render_angles):
        self.root_directory = root_directory
        self.output_image_path = output_dir
        self.flipped = None
        self.render_angles = render_angles

    def render_obj_files(self, config=""):
        if config == "Bellus":
            headscans = self.collect_obj_files_bellus()
            self.flipped = False
        elif config == "FaceVerse":
            headscans = self.collect_obj_files_faceverse()
            self.flipped = True
        elif config == "nphm":
            headscans = self.collect_obj_files_nphm()
            self.flipped = False
        elif config == "facewarehouse":
            headscans = self.collect_obj_files_facewarehouse()
            self.flipped = False
        elif config == "mononphm":
            headscans = self.collect_obj_files_mononphm()
            self.flipped = False
        elif config == "facescape":
            headscans = self.collect_obj_files_facescape()
            self.flipped = False
        else:
            raise AttributeError('No Dataset selected')

        for i in tqdm(headscans, position=0, desc="Render Face"):
            self.render_obj_to_image(i)

    def render_obj_to_image(self, headscan):
        render(self.output_image_path, headscan, self.flipped, self.render_angles)

    def collect_obj_files_bellus(self):
        headscans_paths = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('headscan.obj'):
                    obj_file_path = os.path.join(root, file)
                    splited_path = Path(obj_file_path).parts
                    headscans = {'date': splited_path[-4],
                                 'user': splited_path[-3],
                                 'scan_id': splited_path[-2],
                                 'scan_name': 'headscan',
                                 'obj_file_path': obj_file_path}
                    headscans_paths.append(headscans)

        print('Collected:', len(headscans_paths), 'headscans')
        return headscans_paths

    def collect_obj_files_faceverse(self):
        headscans_paths = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('.obj'):
                    obj_file_path = os.path.join(root, file)
                    splited_path = Path(obj_file_path).parts
                    headscans = {'date': splited_path[-4],
                                 'user': splited_path[-3],
                                 'scan_id': splited_path[-2],
                                 'scan_name': 'headscan',
                                 'obj_file_path': obj_file_path}
                    headscans_paths.append(headscans)

        print('Collected:', len(headscans_paths), 'headscans')
        time.sleep(1)
        return headscans_paths

    def collect_obj_files_facescape(self):

        excluded_paths = {
            "1.1.2020\\66\\models_reg\\3_mouth_stretch.obj",
            "1.1.2020\\148\\models_reg\\13_lip_funneler.obj",
            "1.1.2020\\148\\models_reg\\14_sadness.obj",
            "1.1.2020\\148\\models_reg\\15_lip_roll.obj",
            "1.1.2020\\148\\models_reg\\16_grin.obj",
            "1.1.2020\\148\\models_reg\\17_cheek_blowing.obj",
            "1.1.2020\\148\\models_reg\\18_eye_closed.obj",
            "1.1.2020\\148\\models_reg\\19_brow_raiser.obj",
            "1.1.2020\\148\\models_reg\\20_brow_lower.obj",
            "1.1.2020\\169\\models_reg\\10_dimpler.obj",
            "1.1.2020\\169\\models_reg\\16_grin.obj",
            "1.1.2020\\210\\models_reg\\2_smile.obj",
            "1.1.2020\\323\\models_reg\\11_chin_raiser.obj",
            # "1.1.2020\\433\\models_reg\\12_lip_puckerer.obj",
            "1.1.2020\\452\\models_reg\\18_eye_closed.obj",
            "1.1.2020\\488\\models_reg\\8_mouth_left.obj",
            # "1.1.2020\\501\\models_reg\\10_dimpler.obj",
            "1.1.2020\\510\\models_reg\\8_mouth_left.obj",
            "1.1.2020\\522\\models_reg\\3_mouth_stretch.obj",
            "1.1.2020\\554\\models_reg\\7_jaw_forward.obj",
            "1.1.2020\\554\\models_reg\\18_eye_closed.obj",
            "1.1.2020\\655\\models_reg\\13_lip_funneler.obj",
            "1.1.2020\\655\\models_reg\\20_brow_lower.obj",
            "1.1.2020\\657\\models_reg\\9_mouth_right.obj",
            "1.1.2020\\696\\models_reg\\5_jaw_left.obj",
            "1.1.2020\\726\\models_reg\\16_grin.obj",
            "1.1.2020\\730\\models_reg\\10_dimpler.obj",
            "1.1.2020\\730\\models_reg\\16_grin.obj",
            "1.1.2020\\730\\models_reg\\18_eye_closed.obj",
            "1.1.2020\\731\\models_reg\\5_jaw_left.obj",
            "1.1.2020\\733\\models_reg\\5_jaw_left.obj",
            "1.1.2020\\834\\models_reg\\2_smile.obj",
        }

        headscans_paths = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('.obj'):
                    obj_file_path = os.path.join(root, file)

                    rel_path = os.path.relpath(obj_file_path, self.root_directory)

                    if rel_path in excluded_paths:
                        continue

                    splited_path = Path(obj_file_path).parts
                    headscans = {'date': splited_path[-4],
                                 'user': splited_path[-3],
                                 'scan_id': splited_path[-2],
                                 'scan_name': Path(file).stem,
                                 'obj_file_path': obj_file_path}
                    headscans_paths.append(headscans)

        print('Collected:', len(headscans_paths), 'headscans')
        return headscans_paths

    def collect_obj_files_nphm(self):
        headscans_paths = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('scan.ply'):
                    obj_file_path = os.path.join(root, file)
                    splited_path = Path(obj_file_path).parts
                    headscans = {'date': splited_path[-4],
                                 'user': splited_path[-3],
                                 'scan_id': splited_path[-2],
                                 'scan_name': 'headscan',
                                 'obj_file_path': obj_file_path,
                                 'texture_path': obj_file_path.replace('scan.ply', 'texture.png')}
                    headscans_paths.append(headscans)

        print('Collected:', len(headscans_paths), 'headscans')
        return headscans_paths

    def collect_obj_files_mononphm(self):
        headscans_paths = []
        for root, _, files in tqdm(os.walk(self.root_directory), desc="Search render jobs"):
            for file in files:
                if file.endswith('mesh.obj'):
                    obj_file_path = os.path.join(root, file)
                    splited_path = Path(obj_file_path).parts
                    headscans = {'date': splited_path[-4],
                                 'user': splited_path[-3],
                                 'scan_id': splited_path[-2],
                                 'scan_name': 'headscan',
                                 'obj_file_path': obj_file_path}
                    headscans_paths.append(headscans)

        print('Collected:', len(headscans_paths), 'headscans')
        return headscans_paths

    def collect_obj_files_facewarehouse(self):
        headscans_paths = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('.obj') and "pose" in file:
                    obj_file_path = os.path.join(root, file)
                    splited_path = Path(obj_file_path).parts
                    headscans = {'date': splited_path[-4],
                                 'user': splited_path[-3],
                                 'scan_id': splited_path[-2],
                                 'scan_name': 'headscan',
                                 'obj_file_path': obj_file_path}
                    headscans_paths.append(headscans)

        print('Collected:', len(headscans_paths), 'headscans')
        return headscans_paths
