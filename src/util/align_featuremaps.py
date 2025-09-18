import torch
import torch.nn.functional as F


def align_featuremap(featuremap, grid):
    C, H, W = featuremap.shape
    input_tensor = featuremap.unsqueeze(0)

    grid = grid.unsqueeze(0).to("cuda")
    grid = F.interpolate(grid.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True)
    grid = grid.permute(0, 2, 3, 1)[..., [1, 0]].to(input_tensor.device).float()

    warped_tensor = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True)
    return warped_tensor.squeeze(0)


def align_featuremaps(featuremaps, face_corr, zero_position, device="cuda"):
    batch_size, num_views, num_channels, h, w = featuremaps.shape
    aligned = torch.empty((batch_size, num_views, num_channels, h, w), dtype=featuremaps.dtype, device=device)

    for b in range(batch_size):
        for v in range(num_views):
            if v == zero_position or v >= face_corr[b].shape[0]:
                aligned[b, v] = featuremaps[b, v]
            else:
                aligned[b, v] = align_featuremap(featuremaps[b, v], face_corr[b][v])
    return aligned
