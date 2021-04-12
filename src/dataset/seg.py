import torch
import PIL.Image
import pathlib
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.arg.args import args


class VocDataset(Dataset):
    class_names = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ]

    def __init__(self, split='train', crop_size=(320, 320)):
        assert split in ["train", "val"]
        self.crop_size = crop_size

        self.imgset_file_path = pathlib.Path(args.raw_data_dir) / ('ImageSets/Segmentation/%s.txt' % split)
        self.img_root_dir = pathlib.Path(args.raw_data_dir) / "JPEGImages"
        self.lbl_root_dir = pathlib.Path(args.raw_data_dir) / "SegmentationClass"

        self.total_path = self.get_total_path()

        super(VocDataset, self).__init__()

    def check_img_size(self, img_path):
        min_size = self.crop_size

        img = PIL.Image.open(img_path)
        img_width, img_height = img.size
        return img_width > min_size[0] and img_height > min_size[1]

    def get_total_path(self, ):
        result = []
        with open(self.imgset_file_path) as f:
            for img_id in f:
                img_id = img_id.strip()
                img_path = self.img_root_dir / ('%s.jpg' % img_id)
                lbl_path = self.lbl_root_dir / ('%s.png' % img_id)
                if self.check_img_size(img_path):
                    result.append({
                        'img_path': img_path,
                        'lbl_path': lbl_path,
                    })

        return result

    def __len__(self):
        return len(self.total_path)
    

    def __getitem__(self, index):
        img_path = self.total_path[index]["img_path"]
        lbl_path = self.total_path[index]["lbl_path"]
        img = PIL.Image.open(img_path)
        lbl = PIL.Image.open(lbl_path)
        
        img, lbl = self.center_crop(img, lbl, self.crop_size)

        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                
        lbl = np.array(lbl).astype(np.float32)
        lbl_tensor = torch.from_numpy(lbl).float()

        return img_tensor, lbl_tensor

    def center_crop(self, img, lbl, crop_size):
        w, h = img.size
        size = crop_size[0]
        x1 = int(round((w - size) / 2.))
        y1 = int(round((h - size) / 2.))
        img = img.crop((x1, y1, x1 + size, y1 + size))
        lbl = lbl.crop((x1, y1, x1 + size, y1 + size))

        return img, lbl


if __name__ == "__main__":
    pass



