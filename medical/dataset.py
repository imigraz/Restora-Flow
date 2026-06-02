import numpy as np
import torch
import os

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from xray_dataset_augmentation import XRayDatasetAugmentation


class XrayHand38ChannelsImageGenerator(Dataset):
    def __init__(self, input_size, heatmap_sigma, experiment_name, train=True):
        xray_hand_folder = os.path.join('dataset', 'xray_hand')
        image_extension = '.mhd'

        self.w = input_size
        self.h = input_size
        self.train = train
        self.transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
        self.num_landmarks = 37

        image_ids_file = os.path.join(xray_hand_folder, 'setup', experiment_name, 'train.txt')

        if not train:
            image_ids_file = os.path.join(xray_hand_folder, 'setup', experiment_name, 'test.txt')

        print(image_ids_file)

        with open(image_ids_file) as file:
            self.image_files = [os.path.join(line.rstrip() + image_extension) for line in file]

        self.augmentor = XRayDatasetAugmentation(dataset_folder_path=xray_hand_folder,
                                                 image_extension=image_extension,
                                                 train=train,
                                                 image_ids_file=image_ids_file,
                                                 image_size=[input_size, input_size],
                                                 heatmap_sigma=heatmap_sigma,
                                                 num_landmarks=self.num_landmarks)

    def __getitem__(self, index):
        image_id = self.image_files[index].split('.')[0]

        reference_image, transformation, transformed_image, corresponding_heatmaps, transformed_landmarks = \
            self.augmentor.get_data(image_id)

        # normalization
        transformed_image = ((transformed_image - np.min(transformed_image)) /
                                 (np.max(transformed_image) - np.min(transformed_image)))

        normalized_corresponding_heatmaps = torch.zeros(self.num_landmarks, self.w, self.h).numpy()

        for idx in range(0, self.num_landmarks):
            corresponding_heatmap = corresponding_heatmaps[idx]
            denominator = np.max(corresponding_heatmap) - np.min(corresponding_heatmap)
            if denominator == 0:
                denominator = 0.0000001
            normalized_corresponding_heatmaps[idx] = ((corresponding_heatmap - np.min(corresponding_heatmap)) /
                                                      denominator)

        tensor_image = self.transform(transformed_image)
        tensor_image = tensor_image.swapaxes(0, 1)
        tensor_image = tensor_image.swapaxes(1, 2)

        tensor_corresponding_heatmaps = []
        for idx in range(0, self.num_landmarks):
            tensor_corresponding_heatmaps.append(self.transform(normalized_corresponding_heatmaps[idx]))

        tensor_image_and_heatmap = torch.cat([tensor_image] + tensor_corresponding_heatmaps)
        return tensor_image_and_heatmap.to(dtype=torch.float32), image_id, transformed_landmarks

    def __len__(self):
        return len(self.image_files)
