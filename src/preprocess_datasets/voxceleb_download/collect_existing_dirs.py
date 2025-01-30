import os

def scan_folder(folder_path):
    items = os.listdir(folder_path)
    return items

def save_to_file(file_list, output_file):
    with open(output_file, "w") as f:
        for item in file_list:
            f.write(item + "\n")

if __name__ == '__main__':
    # Example usage
    folder_path = "./your_folder"  # Change this to your desired folder path
    file_list = scan_folder(folder_path)
    output_file = "vox_celeb_output.txt"  # Output file name
    print(len(file_list))
    save_to_file(file_list, output_file)
    print(f"File list saved to {output_file}")