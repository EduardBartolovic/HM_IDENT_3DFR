from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional
from tqdm import tqdm

from face_detection import FaceAligner


# --- Global variable for worker processes ---
worker_aligner: Optional[FaceAligner] = None


def init_worker(model_path: str, alignment_method, device: str):
    """
    Initializes the FaceAligner once per worker process.
    """
    global worker_aligner
    # Initialize with the specific device passed from main
    try:
        # Assuming your FaceAligner accepts device_type and alignment_method in __init__
        # If not, you may need to set them as properties after initialization like:
        # worker_aligner = FaceAligner(model_path, device_type=device)
        # worker_aligner.alignment_method = alignment_method
        worker_aligner = FaceAligner(model_path, device_type=device, alignment_method=alignment_method)
    except TypeError:
        # Fallback if __init__ doesn't support those args directly (common in some implementations)
        worker_aligner = FaceAligner(model_path)
        if hasattr(worker_aligner, 'device_type'):
            worker_aligner.device_type = device
        worker_aligner.alignment_method = alignment_method


def process_single_folder(data_dir: Path, output_root: Path, batch_size: int):
    """
    Worker task: Processes images in a single folder using the pre-loaded global aligner.
    """
    global worker_aligner

    if worker_aligner is None:
        raise RuntimeError("Worker aligner not initialized.")

    folder_name = data_dir.name
    output_path = output_root / folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = list(data_dir.glob("*.jpg"))

    if not image_paths:
        return f"Skipped {folder_name} (No images found)"

    worker_aligner.align_faces(
        [str(p) for p in image_paths],
        batch_size=batch_size,
        debug_folder=str(output_path)
    )

    return f"Processed {folder_name}: {len(image_paths)} images"


def run_batch_alignment(
        data_folders: List[Path],
        model_path: str,
        align_method,
        batch_size: int,
        output_dir: Path,
        num_processes: int,
        device: str = "cpu"
):
    """
    Orchestrates the parallel processing.
    """
    print(f"Starting processing on device: {device}")

    with ProcessPoolExecutor(
            max_workers=num_processes,
            initializer=init_worker,
            initargs=(model_path, align_method, device)
    ) as executor:

        future_to_folder = {
            executor.submit(process_single_folder, folder, output_dir, batch_size): folder
            for folder in data_folders
        }

        for future in tqdm(as_completed(future_to_folder), total=len(data_folders), desc="Aligning Folders"):
            folder = future_to_folder[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Folder {folder.name} generated an exception: {exc}")


if __name__ == "__main__":

    MODEL_FILE = Path('mobile0.25.onnx')
    DATASET_ROOT = Path(r'F:/Face/data/dataset14/test_rgb_bff/enrolled')
    OUTPUT_DIR = Path(r'F:/Face/data/dataset14/BOUNDING_BOX_SCALING/enrolled')

    # Select Device Here ("cpu" or "cuda")
    DEVICE = "cpu"

    folder_paths = [p for p in DATASET_ROOT.iterdir() if p.is_dir()]

    run_batch_alignment(
        data_folders=folder_paths,
        model_path=str(MODEL_FILE),
        align_method=FaceAligner.AlignmentMethod.AURA_FROM_BOUNDING_BOX_SCALING,
        batch_size=32,
        output_dir=OUTPUT_DIR,
        num_processes=4,
        device=DEVICE
    )
