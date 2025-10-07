import cv2

from src.preprocess_datasets.headPoseEstimation.utils.general import draw_axis
from src.preprocess_datasets.preprocess_video import get_frames


def make_video_from_txt(txt_file_path, video_file_path, output_video_path, fix_rotation=True, draw=True):
    with open(txt_file_path, 'r') as txt_file:
        frame_infos = [line.strip().split(',') for line in txt_file.readlines()]

    imgs = get_frames(video_file_path)
    fps = 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frames_written = 0
    for frame, info in zip(imgs, frame_infos):
        y_pred_deg, p_pred_deg, r_pred_deg = map(float, info[:3])
        w, h, c = frame.shape

        if fix_rotation:
            bbox_center_x = w // 2
            bbox_center_y = h // 2
            rotation_matrix = cv2.getRotationMatrix2D((bbox_center_x, bbox_center_y), r_pred_deg, 1.0)

            frame = cv2.warpAffine(
                frame,
                rotation_matrix,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            r_pred_deg = 0

        if draw:
            draw_axis(
                frame,
                y_pred_deg,
                p_pred_deg,
                r_pred_deg,
                bbox=[0, 0, w, h],
                size_ratio=0.5
            )

        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        out.write(frame)
        frames_written += 1

    if out:
        out.release()
        print(f"Processed and saved cropped video: {output_video_path} frames_written: {frames_written}")


if __name__ == '__main__':
    txt_file_path = "E:\\Download\\vox2_mp4_6\\dev\\mp42\\id00027\\4H8NO-Ka_cs\\hpe\\hpe.txt"
    video_file_path = "E:\\Download\\vox2_mp4_6\\dev\\mp42\\id00027\\4H8NO-Ka_cs\\00004.mp4"
    output_video_path = "E:\\Download\\vox2_mp4_6\\dev\\mp42\\id00027\\4H8NO-Ka_cs\\00004-draw.mp4"
    make_video_from_txt(txt_file_path, video_file_path, output_video_path)