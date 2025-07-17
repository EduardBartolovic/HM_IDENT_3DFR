import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_feature_maps(feature_maps, output_dir, stage_names=None, cols=8, batch_idx=0, view_idx=0):
    """
    Visualize and save feature maps from different stages of a neural network.

    Args:
        feature_maps (list or dict): Feature maps to visualize. If a list, each entry is a tensor.
                                     If a dict, keys are stage names and values are tensors.
        output_dir (str): Directory to save the visualizations.
        stage_names (list, optional): Names for the stages (used if feature_maps is a list).
        cols (int): Number of columns for the grid layout.
        batch_idx (int): Index of the sample in the batch to visualize.
        view_idx (int): Index of the sample in the view to visualize.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure feature_maps is a dict for consistency
    if isinstance(feature_maps, list):
        if stage_names is None:
            stage_names = [f"Stage_{i}" for i in range(len(feature_maps))]
        feature_maps = {name: fmap for name, fmap in zip(stage_names, feature_maps)}

    for stage, fmap in feature_maps.items():
        if not (stage == "Stage_0" or stage == "Stage_1" or stage == "Stage_2" or stage == "Stage_3" or stage == "Stage_4"):
            continue

        fmap = fmap[view_idx][batch_idx].detach().cpu().numpy()  # Select batch_idx and convert to numpy
        num_channels = fmap.shape[0]  # Get the number of channels
        rows = int(np.ceil(num_channels / cols))

        print(f"Saving visualizations for {stage}: {fmap.shape} Sample {batch_idx} view_idx {view_idx}")

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  # Increase individual subplot size
        axes = axes.flatten()

        for i in range(num_channels):
            if i < len(axes):
                axes[i].imshow(fmap[i, :, :], cmap='gray', interpolation='nearest')
                axes[i].axis('off')
                axes[i].set_title(f"Ch {i + 1}", fontsize=8)

        for i in range(num_channels, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"{stage} - Sample {batch_idx}", fontsize=8)
        plt.tight_layout(pad=1.0)  # Add padding between subplots

        # Save the figure to the output directory
        output_path = os.path.join(output_dir, f"{stage}_sample_{batch_idx}_{view_idx}.png")
        plt.savefig(output_path, dpi=300)  # Save with higher resolution
        plt.close(fig)

        print(f"Visualization saved to {output_path}")


def plot_feature_maps_grid(tensor, output_path, cols=8, title_prefix=""):
    """
    Plots a grid of feature maps from a tensor of shape (channels, H, W)
    """
    num_channels = tensor.shape[0]
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(num_channels):
        axes[i].imshow(tensor[i], cmap='gray', interpolation='nearest')
        axes[i].axis('off')
        axes[i].set_title(f"Ch {i + 1}", fontsize=8)

    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"{title_prefix}", fontsize=10)
    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def visualize_all_views_and_pooled(all_view_stage, views_pooled_stage, output_dir, sample_idx=1):
    os.makedirs(output_dir, exist_ok=True)

    all_views_sample = all_view_stage[sample_idx]  # shape (25, 64, 112, 112)
    pooled_sample = views_pooled_stage[sample_idx]  # shape (64, 112, 112)

    # Save each view's feature maps
    for view_idx in range(all_views_sample.shape[0]):
        fmap = all_views_sample[view_idx].detach().cpu().numpy()
        output_path = os.path.join(output_dir, f"view_{view_idx:02d}_sample_{sample_idx}.png")
        plot_feature_maps_grid(fmap, output_path, title_prefix=f"View {view_idx}")

    # Save pooled view feature maps
    pooled_fmap = pooled_sample.detach().cpu().numpy()
    output_path = os.path.join(output_dir, f"pooled_sample_{sample_idx}.png")
    plot_feature_maps_grid(pooled_fmap, output_path, title_prefix="Pooled View")

    print(f"All visualizations saved to {output_dir}")