import os
import re
import shutil

def clean_folder(folder_path):
    pattern = re.compile(r"_s\d+_")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if pattern.search(file):
                os.remove(os.path.join(root, file))
                print(f"Removed: {file}")
            else:
                print(f"Kept: {file}")

clean_folder("https://drive.google.com/drive/folders/1fOsbvfqVE_WgBOrmbwO33Pj6RPBFkiRb")
