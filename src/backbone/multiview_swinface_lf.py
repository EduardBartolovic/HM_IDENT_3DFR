import torch
import torch.nn as nn

from src.backbone.swinface.swinface import build_swinface_model


class MultiviewSwinLF(nn.Module):

    def __init__(self, device, aggregator, backbone_fn):
        super().__init__()

        fam_kernel_size = 3
        fam_in_chans = 2112
        fam_conv_shared = False
        fam_conv_mode = "split"
        fam_channel_attention = "CBAM"
        fam_spatial_attention = None
        fam_pooling = "max"
        fam_la_num_list = [2 for j in range(11)]
        fam_feature = "all"
        fam = "3x3_2112_F_s_C_N_max"
        embedding_size = 512

        self.backbone_reg = backbone_fn(embedding_size, fam_in_chans, fam_kernel_size, fam_conv_shared, fam_conv_mode, fam_channel_attention, fam_spatial_attention, fam_pooling, fam_la_num_list, fam_feature)
        self.aggregator = aggregator
        self.device = device

    def forward(self, inputs, perspectives, face_corr, use_face_corr):
        """
        inputs: list of every view: [(B,C,H,W), (B,C,H,W), (B,C,H,W), ...]

        output:
            embeddings_reg: (V, B, 512)
            embeddings_agg  (B, 512)
        """
        with torch.no_grad():
            all_views_embeddings = []
            for view in inputs:
                output_dict = self.backbone_reg(view.to(self.device))
                embeddings = output_dict["Recognition"]
                all_views_embeddings.append(embeddings)

        embeddings_concat = torch.stack(all_views_embeddings, dim=1)  # [B, V, dim]
        embeddings_agg = self.aggregator(embeddings_concat)
        embeddings_reg = all_views_embeddings

        return embeddings_reg, embeddings_agg

    def eval(self):
        self.backbone_reg.eval()
        self.aggregator.eval()


def swinface_mv_lf(device, aggregators):
    return MultiviewSwinLF(device, aggregators, build_swinface_model)

