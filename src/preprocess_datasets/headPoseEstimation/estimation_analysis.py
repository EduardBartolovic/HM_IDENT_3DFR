import os
import matplotlib.pyplot as plt
import seaborn as sns


def process_txt_and_analyze_angles(txt_root_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    total_frame_numbers = []
    total_yaw_angles = []
    total_pitch_angles = []
    total_roll_angles = []
    widths = []
    heights = []

    for root, _, files in os.walk(txt_root_folder):
        for txt_file in files:
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(root, txt_file)

                frame_numbers = []
                yaw_angles = []
                pitch_angles = []
                roll_angles = []

                with open(txt_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        frame_info = list(map(int, line.strip().split(',')))
                        frame_number, x_min, y_min, x_max, y_max = frame_info[:5]
                        y_pred_deg, p_pred_deg, r_pred_deg = frame_info[5:]

                        # Store angles and frame number for plotting
                        frame_numbers.append(frame_number)
                        yaw_angles.append(y_pred_deg)
                        pitch_angles.append(p_pred_deg)
                        roll_angles.append(r_pred_deg)

                    widths.append(x_max - x_min)
                    heights.append(y_max - y_min)

                plotting = False
                if plotting:
                    relative_dir = os.path.relpath(root, txt_root_folder)
                    save_dir = os.path.join(output_folder, relative_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    line_plot_output_path = os.path.join(save_dir,
                                                         f"{os.path.splitext(txt_file)[0]}_angles_line_plot.png")
                    scatter_plot_output_path = os.path.join(save_dir,
                                                            f"{os.path.splitext(txt_file)[0]}_angles_scatter_plot.png")
                    dense_plot_output_path = os.path.join(save_dir,
                                                          f"{os.path.splitext(txt_file)[0]}_angles_dense_plot.png")

                    plt.figure(figsize=(10, 6))
                    plt.plot(frame_numbers, yaw_angles, label='Yaw (°)', color='r', linestyle='-')
                    plt.plot(frame_numbers, pitch_angles, label='Pitch (°)', color='g', linestyle='-')
                    plt.plot(frame_numbers, roll_angles, label='Roll (°)', color='b', linestyle='-')
                    plt.title(f'Viewing Angle Changes (Line Plot) for {txt_file}')
                    plt.xlabel('Frame Number')
                    plt.ylabel('Angle (°)')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(line_plot_output_path)
                    plt.close()

                    plt.figure(figsize=(10, 6))
                    sns.kdeplot(x=yaw_angles, y=pitch_angles, cmap='Reds', fill=True, thresh=0.05, levels=50)
                    plt.scatter(yaw_angles, pitch_angles, color='b', s=10, alpha=0.4)
                    plt.title(f'Dense Plot of Yaw vs Pitch Angles for {txt_file}')
                    plt.xlabel('Yaw Angle (°)')
                    plt.ylabel('Pitch Angle (°)')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(dense_plot_output_path)
                    plt.close()
                    print(f"Saved plot: {scatter_plot_output_path}")

                total_frame_numbers.extend(frame_numbers)
                total_yaw_angles.extend(yaw_angles)
                total_pitch_angles.extend(pitch_angles)
                total_roll_angles.extend(roll_angles)

    print("Number of Videos:", len(widths))
    width_height_plot = os.path.join(output_folder, "width_height_plot.png")
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, color='b', s=10, alpha=0.4)
    plt.title(f'Dense Plot of widths and heights for all')
    plt.xlabel('widths')
    plt.ylabel('heights')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(width_height_plot)
    plt.close()

    yaw_pitch_plot = os.path.join(output_folder, "yaw_pitch_plot.png")
    plt.figure(figsize=(10, 6))
    plt.scatter(total_yaw_angles, total_pitch_angles, color='b', s=10, alpha=0.25)
    # sns.kdeplot(x=total_yaw_angles, y=total_pitch_angles, cmap='Reds', fill=True, thresh=0.05, levels=50)
    plt.title('Dense Plot of Yaw and Pitch Angles for all')
    plt.xlabel('Yaw Angle (°)')
    plt.ylabel('Pitch Angle (°)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(yaw_pitch_plot)
    plt.close()


if __name__ == '__main__':
    txt_folder = "F:\\Face\\HPE\\VoxCeleb1_test_out\\video"  # Folder containing _frame_infos.txt files
    output_folder = "F:\\Face\\HPE\\hpe_analyse"  # Folder to save cropped videos and angle plots

    process_txt_and_analyze_angles(txt_folder, output_folder)
