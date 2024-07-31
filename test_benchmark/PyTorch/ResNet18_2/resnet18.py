from torch import nn
import torch
from torch.nn import functional as F
from resnet import resnet18
# from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, estimate_voting_distribution_with_mean
# from lib.config import cfg
from torchvision.ops import RoIAlign
import math


class Resnet18(nn.Module):

    def __init__(self,
                 ver_dim,
                 seg_dim,
                 normal_dim,
                 fcdim=256,
                 s8dim=128,
                 s4dim=64,
                 s2dim=32,
                 raw_dim=32):
        super(Resnet18, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=False,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        self.norm_dim = normal_dim

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim), nn.ReLU(True))
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim), nn.LeakyReLU(0.1, True))
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim), nn.LeakyReLU(0.1, True))

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim), nn.LeakyReLU(0.1, True))
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x1s -> 32
        self.convraw = nn.Sequential(
            nn.Conv2d(3 + 2 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            # nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )

        self.convraw2 = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            # nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )

        # normal + mask branch
        self.convNormal = nn.Sequential(
            nn.Conv2d(raw_dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, normal_dim + seg_dim, 1, 1))

        self.ActivateNormal = nn.Sequential(nn.BatchNorm2d(3),
                                            nn.LeakyReLU(0.1, True))

        self.ActivateMask = nn.Sequential(nn.BatchNorm2d(2),
                                          nn.LeakyReLU(0.1, True))
        self.convNormal2 = nn.Sequential(
            nn.Conv2d(3 + 2, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True))
        self.convVote = nn.Sequential(
            nn.Conv2d(raw_dim * 2, raw_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim * 2), nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim * 2, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1))

        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        # Freeze first 5 conv layer in ResNet
        # To resolve the domain gap between blender images and real images

        for name, parameter in self.named_parameters():
            if "resnet18_8s.conv1" in name  \
               or "resnet18_8s.bn1" in name \
               or "resnet18_8s.layer1" in name:
                parameter.requires_grad = False

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    # def decode_keypoint(self, output):
    #     vertex = output['vertex'].permute(0, 2, 3, 1)
    #     b, h, w, vn_2 = vertex.shape
    #     vertex = vertex.view(b, h, w, vn_2//2, 2)
    #     mask = torch.argmax(output['seg'], 1)
    #     if cfg.test.un_pnp:
    #         mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
    #         kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
    #         output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
    #     else:
    #         kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
    #         output.update({'mask': mask, 'kpt_2d': kpt_2d})

    def compute_roi(self, hmax, hmin, wmax, wmin, h, w):
        min_y = hmin
        min_x = wmin
        max_y = hmax
        max_x = wmax
        center_x = int(math.floor((min_x + max_x) / 2))
        center_y = int(math.floor((min_y + max_y) / 2))
        half_roi_len = max(abs(center_x - min_x), abs(center_x - max_x),
                           abs(center_y - min_y), abs(center_y - max_y))
        roi_len = int(min(h, w, half_roi_len * 2 + 1))
        half_roi_len = int(math.ceil((roi_len - 1) / 2))
        roi_box = torch.Tensor([
            max(0, center_x - half_roi_len),
            max(0, center_y - half_roi_len),
            min(w - 1, center_x + half_roi_len),
            min(h - 1, center_y + half_roi_len)
        ])

        roi = torch.zeros(4, dtype=torch.int32)
        if roi_box[0] == 0:
            roi[0] = roi_box[0]
        elif roi_box[2] == w - 1:
            roi[0] = w - roi_len
        else:
            roi[0] = roi_box[0]

        if roi_box[1] == 0:
            roi[1] = roi_box[1]
        elif roi_box[3] == h - 1:
            roi[1] = h - roi_len
        else:
            roi[1] = roi_box[1]

        roi[2] = roi_len
        roi[3] = roi_len

        return roi.int()

    def forward(self, x, rois=None, feature_alignment=False):
        x_inp = x[:, 0:3]
        x_mesh = x[:, 3:]
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x_inp)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)
        if fm.shape[2] == 136:
            fm = nn.functional.interpolate(fm, (135, 180),
                                           mode='bilinear',
                                           align_corners=False)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        x1 = self.convraw(torch.cat([fm, x_inp, x_mesh], 1))

        norm_mask = self.convNormal(x1)
        norm_activate = self.ActivateNormal(norm_mask[:, self.norm_dim -
                                                      3:self.norm_dim, :, :])

        mask_activate = self.ActivateMask(norm_mask[:, self.norm_dim:, :, :])
        norm_mask_to_cat = self.convNormal2(
            torch.cat([norm_activate, mask_activate], 1))
        x2 = self.convraw2(torch.cat([fm, x_inp], 1))
        vote_mask = self.convVote(torch.cat([norm_mask_to_cat, x2], 1))

        coarse_seg_pred = norm_mask[:, self.norm_dim:, :, :]
        norm_pred = norm_mask[:, :self.norm_dim, :, :]
        fine_seg_pred = vote_mask[:, self.ver_dim:, :, :]
        ver_pred = vote_mask[:, :self.ver_dim, :, :]

        ret = {
            'coarse_seg': coarse_seg_pred,
            'norm': norm_pred,
            'seg': fine_seg_pred,
            'vertex': ver_pred
        }

        # if not self.training:
        #     with torch.no_grad():
        #         self.decode_keypoint(ret)
        #         if cfg.train.npvnet == 'npvnet_v11':
        #             ret['kpt_2d_ori'] = ret['kpt_2d']
        #             ret['kpt_2d'] = ret['kpt_2d'] / cfg.train.roi_align_size * roi[:, 2].reshape(-1, 1, 1) \
        #                     + roi[:, None, 0:2]

        return ret


def get_res_pvnet(ver_dim, seg_dim, normal_dim):
    model = Resnet18(ver_dim, seg_dim, normal_dim)
    return model
