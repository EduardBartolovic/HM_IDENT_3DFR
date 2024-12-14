import os

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderWithVoting(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Root directory path where data is stored.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.persons = []
        self.numbers = []  # set
        self.ids = []

        # Prepare the image paths and labels
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Traverses the directory structure and collects image paths with their labels.
        """
        person_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if
                       os.path.isdir(os.path.join(self.root_dir, d))]

        for label_idx, person_dir in enumerate(person_dirs):
            person_name = os.path.basename(person_dir)  # This could be used for the label or ID purposes
            subfolders = [os.path.join(person_dir, subdir) for subdir in os.listdir(person_dir) if
                          os.path.isdir(os.path.join(person_dir, subdir))]

            for subfolder in subfolders:
                # Collect all image files in the subfolder
                counter = 0
                for image_name in os.listdir(subfolder):
                    image_path = os.path.join(subfolder, image_name)
                    if image_path.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(image_path)
                        self.labels.append(label_idx)  # Assign a numerical label for each person
                        self.persons.append(person_name)
                        self.numbers.append(int(os.path.basename(subfolder)))  # if of recorded set
                        self.ids.append(counter)  # if of recorded set
                        counter += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image from file
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Convert to RGB if needed
        label = self.labels[idx]
        person = self.persons[idx]
        number = self.numbers[idx]
        image_id = self.ids[idx]

        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, person, number, image_id
