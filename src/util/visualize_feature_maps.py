import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_maps(feature_maps, stage_names=None, cols=8, batch_idx=0):
    """
    Visualize feature maps from different stages of a neural network.

    Args:
        feature_maps (list or dict): Feature maps to visualize. If a list, each entry is a tensor.
                                     If a dict, keys are stage names and values are tensors.
        stage_names (list, optional): Names for the stages (used if feature_maps is a list).
        cols (int): Number of columns for the grid layout.
        batch_idx (int): Index of the sample in the batch to visualize.
    """
    # Ensure feature_maps is a dict for consistency
    if isinstance(feature_maps, list):
        if stage_names is None:
            stage_names = [f"Stage {i+1}" for i in range(len(feature_maps))]
        feature_maps = {name: fmap for name, fmap in zip(stage_names, feature_maps)}

    for stage, fmap in feature_maps.items():
        if stage != "stage_0":
            continue
        fmap = fmap[batch_idx].detach().cpu().numpy()  # Select batch_idx and convert to numpy
        num_channels = fmap.shape[0]                  # Get the number of channels
        rows = int(np.ceil(num_channels / cols))

        print(f"Visualizing {stage}: {fmap.shape} (Sample {batch_idx})")

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()

        for i in range(num_channels):
            if i < len(axes):
                # Display each channel as a grayscale image
                axes[i].imshow(fmap[i, :, :], cmap='gray')
                axes[i].axis('off')
                axes[i].set_title(f"Channel {i+1}")

        for i in range(num_channels, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"{stage} - Sample {batch_idx}", fontsize=16)
        plt.tight_layout()
        plt.show()