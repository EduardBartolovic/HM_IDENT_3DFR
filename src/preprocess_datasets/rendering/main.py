import torch
from pathlib import Path

from src.preprocess_datasets.process_dataset_retinaface import face_crop_and_alignment
from src.preprocess_datasets.rendering import PrepareDataset
from src.preprocess_datasets.rendering.Extract2DFaces import Extract2DFaces
from src.preprocess_datasets.rendering.OBJToRGBD import ObjFileRenderer


def main():

    bellus = True
    facescape = False
    faceverse = False

    texas = False
    facewarehouse = False
    mononphm = False
    ffhq = False
    prep_data = False
    colorferet = False

    bff = False

    root = 'F:\\Face\\data\\dataset14\\'
    render_angles = [-35, -25, -15, -10, -5, 0, 5, 10, 15, 25, 35] #[-25, -10, 0, 10, 25] #  [-10, 0, 10]  #  # [-10, -5, 0, 5, 10]

    # -------- Bellus --------
    if bellus:
        print("################# BELLUS #################")

        # Image Rendering
        directory_path = Path(r'H:\\Maurer\\Bellus')
        output_dir = Path(r"F:\Face\data\tmp5_simulatederror\3D_bellus")
        obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        obj_reader.render_obj_files("Bellus")

        # Photos for 2D
        #directory_path = Path('H:\\Maurer\\Bellus\\')
        #output_dir = Path('F:\\Face\\data\\tmp\\2D_Bellus\\')
        ##obj_reader = Extract2DFaces(directory_path, output_dir)
        #obj_reader.extract_photos()

        # Prepare Dataset Depth:
        #input_path = Path('F:\\Face\\data\\tmp\\3D_Bellus')
        #output_dir = Path(root+'test_depth_bellus')
        #PrepareDataset.prepare_dataset_depth(input_path, output_dir)

        # Prepare Dataset RGB:
        input_path = Path('F:\\Face\\data\\tmp4_283\\3D_Bellus\\')
        output_dir = Path(root+'test_rgb_bellus')
        #PrepareDataset.prepare_dataset_rgb(input_path, output_dir)

        # Prepare Dataset RGB + Depth:
        #input_path = Path(root+'test_rgb_bellus')
        #input_path2 = Path(root+'test_depth_bellus')
        #output_dir = Path(root+'test_rgbd_bellus')
        #PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

        # Prepare Dataset Photos:
        #input_path = Path('F:\\Face\\data\\tmp\\2D_Bellus\\')
        #output_dir = Path(root+'test_photo_bellus')
        #PrepareDataset.prepare_dataset_photos(input_path, output_dir)

    # -------- FACESCAPE --------
    if facescape:
        print("################# FACESCAPE #################")

        # Image Rendering
        directory_path = Path('H:\\Maurer\\facescape\\trainset\\')
        output_dir = Path('F:\\Face\\data\\tmp4_283\\3D_facescape\\')
        obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        obj_reader.render_obj_files('facescape')

        # Prepare Dataset Depth:
        #input_path = Path('F:\\Face\\data\\tmp\\3D_facescape')
        #output_dir = Path(root+'test_depth_facescape')
        #PrepareDataset.prepare_dataset_depth(input_path, output_dir, mode='facescape')

        # Prepare Dataset RGB:
        input_path = Path('F:\\Face\\data\\tmp4_283\\3D_facescape\\')
        output_dir = Path(root+'test_rgb_facescape')
        PrepareDataset.prepare_dataset_rgb(input_path, output_dir, mode='facescape')

        # Prepare Dataset RGB + Depth:
        #input_path = Path(root+'test_rgb_facescape')
        #input_path2 = Path(root+'test_depth_facescape')
        #output_dir = Path(root+'test_rgbd_facescape')
        #PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

    if faceverse:
        print("################# FACEVERSE #################")
        # Image Rendering faceverse
        directory_path = Path('H:\\Maurer\\FaceVerse\\')
        output_dir = Path('F:\\Face\\data\\tmp4_283\\3D_faceverse\\')
        obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        obj_reader.render_obj_files("FaceVerse")

        # Prepare Dataset Depth faceverse:
        #input_path = Path('F:\\Face\\data\\tmp\\3D_faceverse')
        #output_dir = Path(root+'test_depth_faceverse')
        #PrepareDataset.prepare_dataset_depth(input_path, output_dir)

        # Prepare Dataset RGB faceverse:
        input_path = Path('F:\\Face\\data\\tmp4_283\\3D_faceverse\\')
        output_dir = Path(root+'test_rgb_faceverse')
        PrepareDataset.prepare_dataset_rgb(input_path, output_dir)

        # Prepare Dataset RGB + Depth faceverse:
        #input_path = Path(root+'test_rgb_faceverse')
        #input_path2 = Path(root+'test_depth_faceverse')
        #output_dir = Path(root+'test_rgbd_faceverse')
        #PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)


    # Texas
    if texas:
        input_path = Path('H:\\Maurer\\Texas3DFRDatabase\\PreprocessedImages')
        output_dir_rgb = Path(root+'test_rgb_texas')
        output_dir_depth = Path(root+'test_depth_texas')
        output_dir_rgbd = Path(root+'test_rgbd_texas')
        PrepareDataset.prepare_dataset_texas3d(input_path, output_dir_rgb, output_dir_depth)
        PrepareDataset.prepare_dataset_rgbd(output_dir_rgb, output_dir_depth, output_dir_rgbd)

    if mononphm:

        # Image Rendering facescape
        directory_path = Path('H:\\Maurer\\FFHQ-MonoNPHM')
        output_dir = Path('F:\\Face\\data\\tmp\\3D_FFHQMonoNPHM\\')
        # obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        # obj_reader.render_obj_files("mononphm")

        # Prepare Dataset Depth Facescape:
        input_path = Path('F:\\Face\\data\\tmp\\3D_FFHQMonoNPHM')
        output_dir = Path(root+'test_depth_monoffhq')
        # PrepareDataset.prepare_dataset_depth(input_path, output_dir, )

        # Prepare Dataset RGB Facescape:
        input_path = Path('F:\\Face\\data\\tmp\\3D_FFHQMonoNPHM\\')
        output_dir = Path(root+'test_rgb_monoffhq')
        # PrepareDataset.prepare_dataset_rgb(input_path, output_dir)

        # Prepare Dataset RGB + Depth Facescape:
        # input_path = Path(root+'test_rgb_monoffhq')
        # input_path2 = Path(root+'test_depth_monoffhq')
        # output_dir = Path(root+'test_rgbd_monoffhq')
        # PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

        #face_crop_full_frame(root+'rgb_monoffhq70K8', root+'rgb_monoffhq70K_crop8', face_detect_model_root)

    if facewarehouse:

        # Image Rendering facewarehouse
        directory_path = Path('H:\\Maurer\\FaceWarehouse\\data')
        output_dir = Path('F:\\Face\\data\\tmp\\3D_facewarehouse\\')
        obj_reader = ObjFileRenderer(directory_path, output_dir)
        obj_reader.render_obj_files("facewarehouse")

        # Prepare Dataset Depth facewarehouse:
        input_path = Path('F:\\Face\\data\\tmp\\3D_facescape')
        output_dir = Path('F:\\Face\\data\\datasets\\depth_facescape')
        PrepareDataset.prepare_dataset_depth(input_path, output_dir, )

        # Prepare Dataset RGB facewarehouse:
        input_path = Path('F:\\Face\\data\\tmp\\3D_facescape\\')
        output_dir = Path('F:\\Face\\data\\datasets\\rgb_facescape')
        PrepareDataset.prepare_dataset_rgb(input_path, output_dir)

        # Prepare Dataset RGB + Depth facewarehouse:
        input_path = Path('F:\\Face\\data\\datasets\\rgb_facescape')
        input_path2 = Path('F:\\Face\\data\\datasets\\depth_facescape')
        output_dir = Path('F:\\Face\\data\\datasets\\rgbd_facescape')
        PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

    if ffhq:
        directory_paths = [
            Path('ffhq-dataset\\images1024x1024\\00000')
        ]

        # Output directory (common for all)
        output_dir = Path('F:\\Face\\data\\tmp\\2D_ffhq\\')

        # Loop through each directory and process it
        for directory_path in directory_paths:
            obj_reader = Extract2DFaces(directory_path, output_dir)
            obj_reader.gather_photos()

        # Prepare Dataset Photos ffhq:
        input_path = Path('F:\\Face\\data\\tmp\\2D_ffhq\\')
        output_dir = Path(root + 'test_photo_ffhq')
        PrepareDataset.prepare_dataset_photos(input_path, output_dir)

    # COLOR FERET
    if colorferet:

        # Prepare Dataset Photos:
        input_path = Path('H:\\Maurer\\colorferet\\colorferet\\images')
        output_dir = Path(root + 'test_photo_colorferet_1_1')
        #PrepareDataset.prepare_dataset_colorferet_1_1(input_path, output_dir)
        #generate_pairs(output_dir)

        # Prepare Dataset Photos:
        input_path = Path('H:\\Maurer\\colorferet\\colorferet\\images')
        output_dir = Path(root + 'test_photo_colorferet1_n')
        PrepareDataset.prepare_dataset_colorferet_1_n(input_path, output_dir)

    if bff:
        print("################# BFF #################")
        input_paths = [Path(root+'test_rgb_bellus'), Path(root+'test_rgb_facescape'), Path(root+'test_rgb_faceverse')]
        output_dir = Path(root + 'test_rgb_bff')
        #PrepareDataset.prepare_dataset_bff(input_paths, output_dir)

        face_crop_and_alignment(root + 'test_rgb_bff/enrolled', root + 'test_rgb_bff_crop/enrolled', face_factor=0.8, device='cuda' if torch.cuda.is_available() else 'cpu', resize_size=(272, 272))
        face_crop_and_alignment(root + 'test_rgb_bff/query', root + 'test_rgb_bff_crop/query', face_factor=0.8, device='cuda' if torch.cuda.is_available() else 'cpu', resize_size=(272, 272))

        #perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
        #PrepareDataset.filter_views(root + 'test_rgb_bff_crop/enrolled', root + 'test_rgb_bff_crop8/enrolled', perspective_filter, target_views=8)
        #PrepareDataset.filter_views(root + 'test_rgb_bff_crop/query', root + 'test_rgb_bff_crop8/query', perspective_filter, target_views=8)

        #perspective_filter = ['0_0', '0_5', '0_10', '0_15', '0_20', '0_25', '0_30', '0_35', '0_40', '0_45',
        #                      '0_-5', '0_-10', '0_-15', '0_-20', '0_-25', '0_-30', '0_-35', '0_-40', '0_-45',
        #                      '5_0', '10_0', '15_0', '20_0', '25_0', '30_0', '35_0',
        #                      '-5_0', '-10_0', '-15_0', '-20_0', '-25_0', '-30_0', '-35_0',]
        #PrepareDataset.filter_views(root + 'test_rgb_bff/enrolled', root + 'test_rgb_bff_cross/enrolled', perspective_filter, target_views=33)
        #PrepareDataset.filter_views(root + 'test_rgb_bff/query', root + 'test_rgb_bff_cross/query', perspective_filter, target_views=33)

    if prep_data:
        PrepareDataset.prepare_datasets_test(root)
        PrepareDataset.sanity_check(root)


if __name__ == '__main__':
    main()

