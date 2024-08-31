import os
import torch
from PIL import Image
from collections import OrderedDict
from toolkit.dhelper import traverse_recursively
import random
from torch import nn
import numpy as np
import timm
import einops
import torch.nn.functional as F


class SRMConv2d_simple(nn.Module):
    def __init__(self, inc=3):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        self.kernel = torch.from_numpy(self._build_kernel(inc)).float()

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        return filters


class MultiClassificationProcessor_mfolder(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transformer_  = transform
        self.extension_    = '.jpg .jpeg .png .bmp .webp .tif .eps'
        # load category info
        self.ctg_names_    = []             # ctg_idx to ctg_name     
        self.ctg_name2idx_ = OrderedDict()  # ctg_name to ctg_idx
        # load image infos
        self.img_names_  = []    # img_idx to img_name
        self.img_paths_  = []    # img_idx to img_path
        self.img_labels_ = []    # img_idx to img_label

        self.srm = SRMConv2d_simple()

    def load_data_from_dir_test(self, folders):

        # Load image from folder.

        # Args:
        #     dataset_list: dictionary where key is a label and value is a list of folder paths.
        print(folders)
        img_paths = []
        traverse_recursively(folders, img_paths, self.extension_)

        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            self.img_names_.append(img_name)
            self.img_paths_.append(img_path)

        length = len(img_paths)
        print('log: {} image num is {}'.format(folders, length))

    def load_data_from_dir(self, dataset_list):

        # Load image from folder.

        # Args:
        #     dataset_list: dictionary where key is a label and value is a list of folder paths.

        for ctg_name, folders in dataset_list.items():

            if ctg_name not in self.ctg_name2idx_:
                self.ctg_name2idx_[ctg_name] = len(self.ctg_names_)
                self.ctg_names_.append(ctg_name)

            for img_root in folders:
                img_paths = []
                traverse_recursively(img_root, img_paths, self.extension_)

                print(img_root)

                length = len(img_paths)
                for i in range(length):
                    img_path = img_paths[i]
                    img_name = os.path.basename(img_path)
                    self.img_names_.append(img_name)
                    self.img_paths_.append(img_path)
                    self.img_labels_.append(self.ctg_name2idx_[ctg_name])

            print('log: category is %d(%s), image num is %d' % (self.ctg_name2idx_[ctg_name], ctg_name, length))

    def load_data_from_txt(self, img_list_txt, ctg_list_txt):
        """Load image from txt.

        Args:
            img_list_txt: image txt, format is [file_path, ctg_idx].
            ctg_list_txt: category txt, format is [ctg_name, ctg_idx].
        """
        # check
        assert os.path.exists(img_list_txt), 'log: does not exist: {}'.format(img_list_txt)
        assert os.path.exists(ctg_list_txt), 'log: does not exist: {}'.format(ctg_list_txt)

        # load category
        # : open category info file
        with open(ctg_list_txt) as f:
            ctg_infos = [line.strip() for line in f.readlines()] 
        # :load category name & category index
        for ctg_info in ctg_infos:
            tmp      = ctg_info.split(' ')
            ctg_name = tmp[0]
            ctg_idx  = int(tmp[1])
            self.ctg_name2idx_[ctg_name] = ctg_idx
            self.ctg_names_.append(ctg_name)

        # load sample
        # : open image info file
        with open(img_list_txt) as f:
            img_infos = [line.strip() for line in f.readlines()]
        random.shuffle(img_infos)
        # : load image path & category index
        for img_info in img_infos:
            img_path, ctg_name = img_info.rsplit(' ', 1)
            img_name = img_path.split('/')[-1]
            ctg_idx  = int(ctg_name)
            self.img_names_.append(img_name)
            self.img_paths_.append(img_path)
            self.img_labels_.append(ctg_idx)

        for ctg_name in self.ctg_names_:
            print('log: category is %d(%s), image num is %d' % (self.ctg_name2idx_[ctg_name], ctg_name, self.img_labels_.count(self.ctg_name2idx_[ctg_name])))

    def _add_new_channels_worker(self, image):
        new_channels = []

        image = einops.rearrange(image, "h w c -> c h w")
        image = (image- torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN).view(-1, 1, 1)) / torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_STD).view(-1, 1, 1)
        srm = self.srm(image.unsqueeze(0)).squeeze(0)
        new_channels.append(einops.rearrange(srm, "c h w -> h w c").numpy())

        new_channels = np.concatenate(new_channels, axis=2)
        return torch.from_numpy(new_channels).float()

    def add_new_channels(self, images):
        images_copied = einops.rearrange(images, "c h w -> h w c")
        new_channels = self._add_new_channels_worker(images_copied)
        images_copied = torch.concatenate([images_copied, new_channels], dim=-1)
        images_copied = einops.rearrange(images_copied, "h w c -> c h w")

        return images_copied

    def __getitem__(self, index):
        img_path = self.img_paths_[index]

        img_data = Image.open(img_path).convert('RGB')
        img_size = img_data.size[::-1]   # [h, w]

        all_data = []
        for transform in self.transformer_:
            current_data = transform(img_data)
            current_data = self.add_new_channels(current_data)
            all_data.append(current_data)
        img_label = self.img_labels_[index]

        return torch.stack(all_data, dim=0), img_label, img_path, img_size

    def __len__(self):
        return len(self.img_names_)
