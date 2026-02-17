import os
import random
from itertools import combinations


def generate_pairs(dataset_dir, num_negatives=None, output_file='pairs.txt'):
    positive_pairs = []
    negative_pairs = []

    # Get all person directories
    person_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if
                   os.path.isdir(os.path.join(dataset_dir, d))]

    # Generate positive pairs (same person)
    for person_dir in person_dirs:
        person_name = os.path.basename(person_dir)  # Extract the person's name
        # Get subfolders for each person (these are the "folders" you're pairing)
        folders = [os.path.basename(f) for f in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, f))]

        # Create all combinations of folder pairs within the same person (positive pairs)
        for pair in combinations(folders, 2):
            positive_pairs.append(f"{person_name}\t{pair[0]}\t{pair[1]}")

    # Generate negative pairs (different people)
    all_folders = [(os.path.basename(person_dir), os.path.basename(folder)) for person_dir in person_dirs for folder in
                   os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, folder))]

    if num_negatives is None:
        num_negatives = len(positive_pairs)  # Default: Create as many negative pairs as positive ones

    # Create random negative pairs (different people)
    for _ in range(num_negatives):
        person1, folder1 = random.choice(all_folders)
        person2, folder2 = random.choice(all_folders)
        while person1 == person2:  # Ensure the pair is from different people
            person2, folder2 = random.choice(all_folders)

        negative_pairs.append(f"{person1}\t{folder1}\t{person2}\t{folder2}")

    # Write pairs to a .txt file in the required format
    with open(os.path.join(dataset_dir, output_file), 'w') as f:
        # Write positive pairs
        for pair in positive_pairs:
            f.write(f"{pair}\n")
        # Write negative pairs
        for pair in negative_pairs:
            f.write(f"{pair}\n")

    print(f"Pairs written to {os.path.join(dataset_dir, output_file)}")


# Example usage
#dataset_dir = 'F:\\Face\\data\\datasets6\\test_colorferet'  # Replace with the actual dataset path
#generate_pairs(dataset_dir, output_file='pairs.txt')
