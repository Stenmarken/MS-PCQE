import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torchvision import transforms

import torch
from torch.utils import data
import random
import utils_image as util
import open3d as o3d
import matplotlib.pyplot as plt




def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist



def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class VideoDataset_NR_image_with_fast_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, transformations_mask, sampling_div, data_dir_10, data_dir_05, datainfo_path, transform, crop_size, mode_idx):
        super(VideoDataset_NR_image_with_fast_features, self).__init__()
                                        
        # column_names = ['vid_name', 'scene', 'dis_type_level']
        dataInfo = pd.read_csv(datainfo_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.moss = dataInfo['mos']

        self.crop_size = crop_size
        self.data_dir_10 = data_dir_10
        self.data_dir_05 = data_dir_05
        self.sampling_div = sampling_div
        self.transform = transform
        self.transform_mask = transformations_mask
        self.length = len(self.video_names)

        self.mode_idx = mode_idx
        self.user_num = 5  # 1


    def __len__(self):
        if self.mode_idx == 0:
            return self.length * self.user_num
        else:
            return self.length


    def __getitem__(self, idx):


        if self.mode_idx == 0:
            video_name = self.video_names.iloc[idx // self.user_num]
        else:
            video_name = self.video_names.iloc[idx]

        frames_dir_10 = os.path.join(self.data_dir_10, video_name)
        frames_dir_05 = os.path.join(self.data_dir_05,video_name)


        if self.mode_idx == 0:
            mos = self.moss.iloc[idx // self.user_num]
        else:
            mos = self.moss.iloc[idx]
        # mos = torch.Tensor(mos.reshape(1,1))


########################################################################################################################

        video_channel = 3
        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        F_score_big = 1
        # F_score_big = torch.Tensor([F_score_big])
        F_score_small = -1
        # F_score_small = torch.Tensor([F_score_small])


        if self.mode_idx == 0:
            frames_dir_10_read = 1  #   1  self.sampling_div
            frames_dir_05_read = 1  #   1  self.sampling_div
        else:
            frames_dir_10_read = self.sampling_div
            frames_dir_05_read = self.sampling_div


        frames_dir_10_video = torch.zeros([frames_dir_10_read, video_channel, video_height_crop, video_width_crop])
        frames_dir_05_video = torch.zeros([frames_dir_05_read, video_channel, video_height_crop, video_width_crop])
        frames_dir_10_video_mask = torch.zeros([frames_dir_10_read, 1, video_height_crop, video_width_crop])
        frames_dir_05_video_mask = torch.zeros([frames_dir_05_read, 1, video_height_crop, video_width_crop])



        if self.mode_idx == 0:
            j_video = random.randint(1, 6)

        for j in range(frames_dir_10_read):
            if self.mode_idx == 0:
                i_video = random.randint(0, 3)  # 旋转随机

                imge_name_10 = os.path.join(frames_dir_10, str(j_video + j).zfill(3) + '.png')
                imge_name_05 = os.path.join(frames_dir_05, str(j_video + j).zfill(3) + '.png')


                # imge_name_10 = os.path.join(frames_dir_10, str(j+1).zfill(3) + '.png')
                # imge_name_05 = os.path.join(frames_dir_05, str(j+1).zfill(3) + '.png')


                read_frame_10 = Image.open(imge_name_10)
                read_frame_05 = Image.open(imge_name_05)
                read_frame_10 = read_frame_10.convert('RGB')
                read_frame_05 = read_frame_05.convert('RGB')

                read_frame_10_mask = ImageOps.grayscale(read_frame_10)
                read_frame_05_mask = ImageOps.grayscale(read_frame_05)
                read_frame_10_mask = read_frame_10_mask.point(lambda x: 0 if x > 254 else 1, '1')
                read_frame_05_mask = read_frame_05_mask.point(lambda x: 0 if x > 254 else 1, '1')

                # plt.imshow(read_frame_10_mask, cmap='gray')
                # plt.show()


                read_frame_10 = util.augment_img(read_frame_10, mode=i_video)
                read_frame_10 = self.transform(read_frame_10)
                frames_dir_10_video[j] = read_frame_10
                read_frame_05 = util.augment_img(read_frame_05, mode=i_video)
                read_frame_05 = self.transform(read_frame_05)
                frames_dir_05_video[j] = read_frame_05


                read_frame_10_mask = util.augment_img(read_frame_10_mask, mode=i_video)
                read_frame_10_mask = self.transform_mask(read_frame_10_mask).float()
                # tensor_mask = read_frame_10_mask.numpy().transpose(1, 2, 0)
                # plt.imshow(tensor_mask)
                # plt.show()
                frames_dir_10_video_mask[j] = read_frame_10_mask

                read_frame_05_mask = util.augment_img(read_frame_05_mask, mode=i_video)
                read_frame_05_mask = self.transform_mask(read_frame_05_mask).float()
                # tensor_mask = read_frame_05_mask.numpy().transpose(1, 2, 0)
                # plt.imshow(tensor_mask)
                # plt.show()
                frames_dir_05_video_mask[j] = read_frame_05_mask

########################################################################################################################


            else:

                imge_name_10 = os.path.join(frames_dir_10, str(j + 1).zfill(3) + '.png')
                imge_name_05 = os.path.join(frames_dir_05, str(j + 1).zfill(3) + '.png')
                read_frame_10 = Image.open(imge_name_10)
                read_frame_05 = Image.open(imge_name_05)
                read_frame_10 = read_frame_10.convert('RGB')
                read_frame_05 = read_frame_05.convert('RGB')

                read_frame_10_mask = ImageOps.grayscale(read_frame_10)
                read_frame_05_mask = ImageOps.grayscale(read_frame_05)
                read_frame_10_mask = read_frame_10_mask.point(lambda x: 0 if x > 254 else 1, '1')
                read_frame_05_mask = read_frame_05_mask.point(lambda x: 0 if x > 254 else 1, '1')


                read_frame_10 = self.transform(read_frame_10)
                frames_dir_10_video[j] = read_frame_10
                read_frame_05 = self.transform(read_frame_05)
                frames_dir_05_video[j] = read_frame_05

                read_frame_10_mask = self.transform_mask(read_frame_10_mask).float()
                frames_dir_10_video_mask[j] = read_frame_10_mask
                read_frame_05_mask = self.transform_mask(read_frame_05_mask).float()
                frames_dir_05_video_mask[j] = read_frame_05_mask



        return frames_dir_10_video, frames_dir_05_video, frames_dir_10_video_mask, frames_dir_05_video_mask, mos







