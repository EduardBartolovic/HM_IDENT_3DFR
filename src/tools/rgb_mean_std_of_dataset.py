import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_mean_std(data_dir):
    # Applying basic transformations
    input_size = [112, 112]
    basic_transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.ToTensor()
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=basic_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in tqdm(loader):
        for i in range(3):  # loop over RGB channels
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    print("Mean: ", mean)
    print("Std: ", std)

    return mean, std


if __name__ == '__main__':
    # Your dataset directory
    data_dir = 'C:\\Users\\Eduard\\Desktop\\datasets\\test_rgb_texas\\validation'
    print(data_dir)
    mean, std = compute_mean_std(data_dir)

    data_dir = 'C:\\Users\\Eduard\\Desktop\\datasets\\test_rgb_bellus\\validation'
    print(data_dir)
    mean, std = compute_mean_std(data_dir)

    data_dir = 'C:\\Users\\Eduard\\Desktop\\datasets\\test_rgb_facescape\\validation'
    print(data_dir)
    mean, std = compute_mean_std(data_dir)

    data_dir = 'C:\\Users\\Eduard\\Desktop\\datasets\\test_rgb_faceverse\\validation'
    print(data_dir)
    mean, std = compute_mean_std(data_dir)


