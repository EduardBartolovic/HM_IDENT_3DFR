from pathlib import Path

from src.preprocess_datasets.rendering import PrepareDataset
from src.preprocess_datasets.rendering.Extract2DFaces import Extract2DFaces
from src.preprocess_datasets.rendering.OBJToRGBD import ObjFileRenderer


def main():

    bellus = False
    facescape = False
    faceverse = False
    texas = False
    nphm = False
    facewarehouse = False
    mononphm = True
    ffhq = False
    prep_data = False
    colorferet = True
    bff = False

    root = 'F:\\Face\\data\\datasets7\\'
    render_angles = [-25, -10, 0, 10, 25] #  [-10, 0, 10]  #  # [-10, -5, 0, 5, 10]

    # -------- Bellus -------------
    if bellus:

        # Image Rendering Bellus
        directory_path = Path('H:\\Maurer\\Bellus\\')
        output_dir = Path('F:\\Face\\data\\tmp\\3D_bellus\\')
        obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        obj_reader.render_obj_files("Bellus")

        # Photos for 2D Bellus
        directory_path = Path('H:\\Maurer\\Bellus\\')
        output_dir = Path('F:\\Face\\data\\tmp\\2D_Bellus\\')
        obj_reader = Extract2DFaces(directory_path, output_dir)
        obj_reader.extract_photos()
        
        # Prepare Dataset Depth Bellus:
        input_path = Path('F:\\Face\\data\\tmp\\3D_Bellus')
        output_dir = Path(root+'test_depth_bellus')
        PrepareDataset.prepare_dataset_depth(input_path, output_dir)

        # Prepare Dataset RGB Bellus:
        input_path = Path('F:\\Face\\data\\tmp\\3D_Bellus\\')
        output_dir = Path(root+'test_rgb_bellus')
        PrepareDataset.prepare_dataset_rgb(input_path, output_dir)

        # Prepare Dataset RGB + Depth Bellus:
        input_path = Path(root+'test_rgb_bellus')
        input_path2 = Path(root+'test_depth_bellus')
        output_dir = Path(root+'test_rgbd_bellus')
        PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

        # Prepare Dataset Photos Bellus:
        input_path = Path('F:\\Face\\data\\tmp\\2D_Bellus\\')
        output_dir = Path(root+'test_photo_bellus')
        PrepareDataset.prepare_dataset_photos(input_path, output_dir)


    # FACESCAPE
    if facescape:

        # Image Rendering facescape
        directory_path = Path('H:\\Maurer\\facescape\\trainset\\')
        output_dir = Path('F:\\Face\\data\\tmp\\3D_facescape\\')
        obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        obj_reader.render_obj_files('facescape')

        # Prepare Dataset Depth Facescape:
        input_path = Path('F:\\Face\\data\\tmp\\3D_facescape')
        output_dir = Path(root+'test_depth_facescape')
        PrepareDataset.prepare_dataset_depth(input_path, output_dir, mode='facescape')

        # Prepare Dataset RGB Facescape:
        input_path = Path('F:\\Face\\data\\tmp\\3D_facescape\\')
        output_dir = Path(root+'test_rgb_facescape')
        PrepareDataset.prepare_dataset_rgb(input_path, output_dir, mode='facescape')

        # Prepare Dataset RGB + Depth Facescape:
        input_path = Path(root+'test_rgb_facescape')
        input_path2 = Path(root+'test_depth_facescape')
        output_dir = Path(root+'test_rgbd_facescape')
        PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

    if faceverse:

        # Image Rendering faceverse
        directory_path = Path('H:\\Maurer\\FaceVerse\\')
        output_dir = Path('F:\\Face\\data\\tmp\\3D_faceverse\\')
        obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        obj_reader.render_obj_files("FaceVerse")

        # Prepare Dataset Depth faceverse:
        input_path = Path('F:\\Face\\data\\tmp\\3D_faceverse')
        output_dir = Path(root+'test_depth_faceverse')
        PrepareDataset.prepare_dataset_depth(input_path, output_dir)

        # Prepare Dataset RGB faceverse:
        input_path = Path('F:\\Face\\data\\tmp\\3D_faceverse\\')
        output_dir = Path(root+'test_rgb_faceverse')
        PrepareDataset.prepare_dataset_rgb(input_path, output_dir)

        # Prepare Dataset RGB + Depth faceverse:
        input_path = Path(root+'test_rgb_faceverse')
        input_path2 = Path(root+'test_depth_faceverse')
        output_dir = Path(root+'test_rgbd_faceverse')
        PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

    if nphm:
        # Image Rendering nphm
        directory_path = Path('H:\\Maurer\\nphm\\')
        output_dir = Path('F:\\Face\\data\\tmp\\3D_nphm\\')
        obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        obj_reader.render_obj_files("nphm")

        return 0 
    
        # Prepare Dataset Depth nphm:
        input_path = Path('F:\\Face\\data\\tmp\\3D_nphm')
        output_dir = Path('F:\\Face\\data\\datasets4\\test_depth_nphm')
        PrepareDataset.prepare_dataset_depth(input_path, output_dir)

        # Prepare Dataset RGB nphm:
        input_path = Path('F:\\Face\\data\\tmp\\3D_nphm\\')
        output_dir = Path('F:\\Face\\data\\datasets4\\test_rgb_nphm')
        PrepareDataset.prepare_dataset_rgb(input_path, output_dir)

        # Prepare Dataset RGB + Depth nphm:
        input_path = Path('F:\\Face\\data\\datasets4\\test_rgb_nphm')
        input_path2 = Path('F:\\Face\\data\\datasets4\\test_depth_nphm')
        output_dir = Path('F:\\Face\\data\\datasets4\\test_rgbd_nphm')
        PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

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
        obj_reader = ObjFileRenderer(directory_path, output_dir, render_angles)
        #obj_reader.render_obj_files("mononphm")

        # Prepare Dataset Depth Facescape:
        input_path = Path('F:\\Face\\data\\tmp\\3D_FFHQMonoNPHM')
        output_dir = Path(root+'test_depth_monoffhq')
        PrepareDataset.prepare_dataset_depth(input_path, output_dir, )

        # Prepare Dataset RGB Facescape:
        input_path = Path('F:\\Face\\data\\tmp\\3D_FFHQMonoNPHM\\')
        output_dir = Path(root+'test_rgb_monoffhq')
        #PrepareDataset.prepare_dataset_rgb(input_path, output_dir)

        # Prepare Dataset RGB + Depth Facescape:
        input_path = Path(root+'test_rgb_monoffhq')
        input_path2 = Path(root+'test_depth_monoffhq')
        output_dir = Path(root+'test_rgbd_monoffhq')
        #PrepareDataset.prepare_dataset_rgbd(input_path, input_path2, output_dir)

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
            Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\00000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\24000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\25000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\26000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\27000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\33000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\34000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\35000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\36000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\37000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\38000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\39000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\40000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\41000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\42000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\43000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\44000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\45000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\46000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\47000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\48000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\49000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\50000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\51000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\52000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\53000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\54000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\55000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\56000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\57000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\58000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\59000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\60000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\61000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\62000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\63000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\64000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\65000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\66000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\67000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\68000'),
            #Path('H:\\Maurer\\ffhq-dataset\\images1024x1024\\69000'),
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

        input_paths = [Path('F:\\Face\\data\\datasets7\\test_rgb_bellus'), Path('F:\\Face\\data\\datasets7\\test_rgb_facescape'), Path('F:\\Face\\data\\datasets7\\test_rgb_faceverse')]
        output_dir = Path(root + 'test_rgb_bff')
        PrepareDataset.prepare_dataset_bff(input_paths, output_dir)

        input_paths = [Path('F:\\Face\\data\\datasets7\\test_rgbd_bellus'), Path('F:\\Face\\data\\datasets7\\test_rgbd_facescape'), Path('F:\\Face\\data\\datasets7\\test_rgbd_faceverse')]
        output_dir = Path(root + 'test_rgbd_bff')
        PrepareDataset.prepare_dataset_bff(input_paths, output_dir)

    if prep_data:
        PrepareDataset.prepare_datasets_test(root)

        #PrepareDataset.sanity_check(root)


if __name__ == '__main__':
    main()

