import os

import torch

from src.util.misc import colorstr


def load_rgbd_backbone_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    # Splitting the checkpoint for RGB and Depth Inputs
    rgb_checkpoint = {}
    depth_checkpoint = {}

    for k, v in checkpoint.items():
        if 'rgb' in k:
            new_key = k.replace('rgb_', '')
            rgb_checkpoint['rgb_' + new_key] = v
            depth_checkpoint['depth_' + new_key] = v
        else:
            rgb_checkpoint['rgb_' + k] = v
            depth_checkpoint['depth_' + k] = v

    model.rgb_body.load_state_dict(rgb_checkpoint, strict=False)
    model.depth_body.load_state_dict(depth_checkpoint, strict=False)

    return model


def load_checkpoint(BACKBONE, HEAD, BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT, rgbd=False):

    if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
        print(colorstr('blue', f"Loading Backbone Checkpoint {BACKBONE_RESUME_ROOT}"))
        if rgbd:
            load_rgbd_backbone_checkpoint(BACKBONE, BACKBONE_RESUME_ROOT)
        else:
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))

        print(colorstr('blue', f"Loading Head Checkpoint {HEAD_RESUME_ROOT}"))
        HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
    elif os.path.isfile(BACKBONE_RESUME_ROOT):
        print(colorstr('blue', f"Loading ONLY Backbone Checkpoint {BACKBONE_RESUME_ROOT}"))
        if rgbd:
            load_rgbd_backbone_checkpoint(BACKBONE, BACKBONE_RESUME_ROOT)
        else:
            state_dict = torch.load(BACKBONE_RESUME_ROOT, weights_only=True)
            try:
                BACKBONE.load_state_dict(state_dict)
            except RuntimeError:
                print("Warning: Changing State dict!")
                state_dict = adapt_state_dict_for_backbone(state_dict)
                BACKBONE.load_state_dict(state_dict)
    else:
        if len(BACKBONE_RESUME_ROOT) > 5 or len(HEAD_RESUME_ROOT) > 5:
            print(colorstr('red', f"You put in a path but there is no checkpoint found at {BACKBONE_RESUME_ROOT} or {HEAD_RESUME_ROOT}. Please Have a Check or Continue to Train from Scratch"))
            raise Exception(colorstr('red', f"You put in a path but there is no checkpoint found at {BACKBONE_RESUME_ROOT} or {HEAD_RESUME_ROOT}. Please Have a Check or Continue to Train from Scratch"))
        print(colorstr('red', f"No Checkpoint Found at {BACKBONE_RESUME_ROOT} and {HEAD_RESUME_ROOT}. Please Have a Check or Continue to Train from Scratch"))


def adapt_state_dict_for_backbone(state_dict, prefix="backbone."):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = prefix + k if not k.startswith(prefix) else k
        new_state_dict[new_key] = v
    return new_state_dict
