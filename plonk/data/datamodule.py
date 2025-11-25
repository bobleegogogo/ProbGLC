import pytorch_lightning as L
from torch.utils.data import DataLoader, random_split
import torch
import time
import webdataset as wds
from torch.utils.data import default_collate
import math
from PIL import Image


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        full_batch_size,
        num_workers,
        eval_batch_size=None,
        num_nodes=1,
        num_devices=1,
        val_proportion=0.1,
    ):
        super().__init__()
        self._builders = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.num_workers = num_workers
        self.collate_fn = dict_collate_fn()
        self.full_batch_size = full_batch_size
        self.train_batch_size = full_batch_size // (num_nodes * num_devices)
        if eval_batch_size is None:
            self.eval_batch_size = self.train_batch_size
            self.full_eval_batch_size = self.full_batch_size
        else:
            self.eval_batch_size = eval_batch_size // (num_nodes * num_devices)
            self.full_eval_batch_size = eval_batch_size
        print(f"Each GPU will receive {self.train_batch_size} images for training")
        print(f"Each GPU will receive {self.eval_batch_size} images for evaluation")
        self.val_proportion = val_proportion
        self.world_size = num_nodes * num_devices

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"]()
            self.val_dataset = self._builders["val"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")
        else:
            self.test_dataset = self._builders["test"]()
            print(f"Test dataset size: {len(self.test_dataset)}")
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        return DataLoader(
             self.train_dataset,
             batch_size=self.train_batch_size,
             shuffle=True,
             pin_memory=False,
             drop_last=True,
             num_workers=self.num_workers,
         )

    def val_dataloader(self):
        return DataLoader(  
             self.val_dataset, 
             batch_size=self.full_batch_size, 
             shuffle=False, 
             pin_memory=False, 
             drop_last=True, 
             num_workers=self.num_workers, 
         ) 

    def test_dataloader(self):
        return DataLoader(  
             self.test_dataset,  
             batch_size=self.full_batch_size, 
             shuffle=False, 
             pin_memory=False, 
             drop_last=True, 
             num_workers=self.num_workers, 
         ) 
    def get_webdataset_length(
        self, dataset, collate_fn, full_batch_size, batch_size, num_workers=0
    ):
        dataset = dataset.compose(
            wds.batched(
                batch_size,
                partial=self.world_size > 1,
                # dict_collate_and_pad(["flan_t5_xl"], max_length=256),
            )
        )
        num_samples = dataset.num_samples
        if self.world_size > 1:
            num_batches = math.ceil(num_samples / full_batch_size)
            num_workers = max(1, num_workers)

            num_worker_batches = math.ceil(num_batches / num_workers)
            num_batches = num_worker_batches * num_workers
            num_samples = num_batches * full_batch_size

            dataset = dataset.with_epoch(num_worker_batches).with_length(
                num_worker_batches
            )
        else:
            num_batches = math.ceil(num_samples / batch_size)

            dataset = dataset.with_epoch(num_batches).with_length(num_batches)
        return dataset, num_batches


def dict_collate_fn():
    def dict_collate(batch):
        output_dict = {}
        if isinstance(batch[0], dict):
            for key in batch[0].keys():
                output_dict[key] = dict_collate([item[key] for item in batch])
        else:
            # Check if the batch contains PIL images
            if isinstance(batch[0], Image.Image):
                output_dict = batch  # Return list of PIL images directly
            else:
                output_dict = default_collate(batch)
        return output_dict

    return dict_collate
