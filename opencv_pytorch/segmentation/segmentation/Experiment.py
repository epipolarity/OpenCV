import cv2
import os

from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import  DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from albumentations import Compose, Normalize, HorizontalFlip, ShiftScaleRotate, HueSaturationValue
from albumentations.pytorch import ToTensorV2

from trainer import Trainer, hooks

from trainer.utils import setup_system
from trainer.configuration import SystemConfig, DatasetConfig, TrainerConfig, OptimizerConfig, DataloaderConfig
from trainer.matplotlib_visualizer import MatplotlibVisualizer

from segmentation.SemSegDataset import SemSegDataset
from segmentation.LinkNet_Model import LinkNet
from segmentation.IntersectionOverUnion import IntersectionOverUnion


# define experiment class
class Experiment:
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        image_dir: str,
        mask_dir: str,
        # init configs
        system_config: SystemConfig = SystemConfig(),
        dataset_config: DatasetConfig = DatasetConfig(),
        dataloader_config: DataloaderConfig = DataloaderConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        existing_model: bool = False
    ):
        # apply system settings
        self.system_config = system_config
        setup_system(system_config)
        # define train dataloader
        self.loader_train = DataLoader(
            # define our dataset
            SemSegDataset(data_path = train_data_path,
                          images_folder = image_dir,
                          masks_folder = mask_dir,
                          num_classes = 11,
                          class_names = ['Background', 'Person', 'Bike', 'Car',
                                         'Drone', 'Boat', 'Animal', 'Obstacle',
                                         'Construction', 'Vegetation', 'Road', 'Sky'],
                          transforms=Compose([
                              HorizontalFlip(),
                              ShiftScaleRotate(
                              shift_limit=0.0625,
                              scale_limit=0.50,
                              rotate_limit=45,
                              border_mode=cv2.BORDER_CONSTANT,
                              value=0,
                              mask_value=11,
                              p=.75
                              ),
                          HueSaturationValue(),
                          Normalize(),
                          ToTensorV2()])),
            batch_size=dataloader_config.batch_size,
            shuffle=True,
            num_workers=dataloader_config.num_workers,
            pin_memory=True
        )

        # define test dataloader
        self.loader_test = DataLoader(
            SemSegDataset(data_path = val_data_path,
                          images_folder = image_dir,
                          masks_folder = mask_dir,
                          num_classes = 12,
                          class_names = ['Background', 'Person', 'Bike', 'Car',
                                         'Drone', 'Boat', 'Animal', 'Obstacle',
                                         'Construction', 'Vegetation', 'Road', 'Sky'],
                          transforms=Compose([Normalize(), ToTensorV2()])),            
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            pin_memory=True
        )

        # define model
#         self.model = SemanticSegmentation(
#             num_classes=self.loader_test.dataset.get_num_classes(), final_upsample=True
#         )
                  
        self.model = LinkNet(num_classes=self.loader_test.dataset.get_num_classes())
        if existing_model:
            model_path = os.path.join(os.getcwd(), 'checkpoints', 'LinkNet_best.pth')
            if not os.path.exists(model_path):
                raise Exception(f"{model_path} was not found")
            self.model.load_state_dict(torch.load(model_path))
        # define loss function as cross-entropy loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.loader_test.dataset.get_num_classes())
        # define metrics function as intersection over union
        self.metric_fn = IntersectionOverUnion(
            num_classes=self.loader_test.dataset.get_num_classes(), reduced_probs=False
        )
        # define optimizer and its params
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay
        )
        # define learning rate scheduler
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )
        # define visualizer
        self.visualizer = MatplotlibVisualizer()
        # self.visualizer = TensorBoardVisualizer()

    # run training
    def run(self, trainer_config: TrainerConfig) -> dict:
        # apply system settings
        setup_system(self.system_config)
        # move training to the chosen device
        device = torch.device(trainer_config.device)
        # send data to chosen device
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        # define trainer
        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            data_getter=itemgetter("image"),
            target_getter=itemgetter("mask"),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("mean_iou"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        # define hook to run after each epoch
        model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_semseg)
        # run the training
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics