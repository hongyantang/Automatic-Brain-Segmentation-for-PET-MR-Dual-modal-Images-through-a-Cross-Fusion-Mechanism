from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *

from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd
)

import numpy as np
from collections import OrderedDict
import glob

def data_loader(args):
    root_pet_dir = args.root_pet
    root_mr_dir = args.root_mr
    dataset = args.dataset

    if dataset == 'pet':
        print('Start to load data from directory: {}'.format(root_pet_dir))
        if args.mode == 'train':
            train_samples = {}
            valid_samples = {}
            ## Input training data
            train_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTr', '*.nii.gz')))
            train_label = sorted(glob.glob(os.path.join(root_pet_dir, 'labelsTr', '*.nii.gz')))
            train_samples['images'] = train_img
            train_samples['labels'] = train_label
            ## Input validation data
            valid_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesVal', '*.nii.gz')))
            valid_label = sorted(glob.glob(os.path.join(root_pet_dir, 'labelsVal', '*.nii.gz')))
            valid_samples['images'] = valid_img
            valid_samples['labels'] = valid_label
            print('Finished loading all training samples from dataset: {}!'.format(dataset))
            return train_samples, valid_samples
        elif args.mode == 'test':
            test_samples = {}
            ## Input inference data
            test_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTs', '*.nii.gz')))
            test_samples['images'] = test_img
            print('Finished loading all inference samples from dataset: {}!'.format(dataset))
            return test_samples
                         
    elif dataset == 'mr':
        print('Start to load data from directory: {}'.format(root_mr_dir))
        if args.mode == 'train':
            train_samples = {}
            valid_samples = {}
            ## Input training data
            train_img = sorted(glob.glob(os.path.join(root_mr_dir, 'imagesTr', '*.nii.gz')))
            train_label = sorted(glob.glob(os.path.join(root_mr_dir, 'labelsTr', '*.nii.gz')))
            train_samples['images'] = train_img
            train_samples['labels'] = train_label
            ## Input validation data
            valid_img = sorted(glob.glob(os.path.join(root_mr_dir, 'imagesVal', '*.nii.gz')))
            valid_label = sorted(glob.glob(os.path.join(root_mr_dir, 'labelsVal', '*.nii.gz')))
            valid_samples['images'] = valid_img
            valid_samples['labels'] = valid_label
            print('Finished loading all training samples from dataset: {}!'.format(dataset))
            return train_samples, valid_samples
        elif args.mode == 'test':
            test_samples = {}
            ## Input inference data
            test_img = sorted(glob.glob(os.path.join(root_mr_dir, 'imagesTs', '*.nii.gz')))
            test_samples['images'] = test_img
            print('Finished loading all inference samples from dataset: {}!'.format(dataset))
            return test_samples
        
    elif dataset == 'pet_mr':
        print('Start to load data from directory: {}, {}'.format(root_pet_dir, root_mr_dir))
        if args.mode == 'train':
            train_samples = {}
            valid_samples = {}
            ## Input training data
            train_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTr', '*.nii.gz')))
            train_mr_img = sorted(glob.glob(os.path.join(root_mr_dir, 'imagesTr', '*.nii.gz')))
            train_label = sorted(glob.glob(os.path.join(root_mr_dir, 'labelsTr', '*.nii.gz')))
            train_samples['images_pet'] = train_pet_img
            train_samples['images_mr'] = train_mr_img
            train_samples['labels'] = train_label
            ## Input validation data
            valid_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesVal', '*.nii.gz')))#imagesVal
            valid_mr_img = sorted(glob.glob(os.path.join(root_mr_dir, 'imagesVal', '*.nii.gz')))#imagesVal
            valid_label = sorted(glob.glob(os.path.join(root_mr_dir, 'labelsVal', '*.nii.gz')))#labelsVal
            valid_samples['images_pet'] = valid_pet_img
            valid_samples['images_mr'] = valid_mr_img
            valid_samples['labels'] = valid_label
            print('Finished loading all training samples from dataset: {}!'.format(dataset))
            return train_samples, valid_samples
        elif args.mode == 'test':
            test_samples = {}
            ## Input inference data
            test_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTs', '*.nii.gz')))
            test_mr_img = sorted(glob.glob(os.path.join(root_mr_dir, 'imagesTs', '*.nii.gz')))
            test_samples['images_pet'] = test_pet_img
            test_samples['images_mr'] = test_mr_img
            print('Finished loading all inference samples from dataset: {}!'.format(dataset))
            return test_samples           

def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None

    if dataset == 'pet':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'mr':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )        

    elif dataset == 'pet_mr':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image_pet", "image_mr", "label"]),
                AddChanneld(keys=["image_pet", "image_mr", "label"]),
                Orientationd(keys=["image_pet", "image_mr", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image_pet"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),                
                ScaleIntensityRanged(
                    keys=["image_mr"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image_pet", "image_mr", "label"], source_key="image_pet"),
                RandCropByPosNegLabeld(
                    keys=["image_pet", "image_mr", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image_pet",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image_pet", "image_mr"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image_pet', 'image_mr', 'label'],
                    mode=('bilinear', 'bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image_pet", "image_mr", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image_pet", "image_mr", "label"]),
                AddChanneld(keys=["image_pet", "image_mr", "label"]),
                Orientationd(keys=["image_pet", "image_mr", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image_pet"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),                
                ScaleIntensityRanged(
                    keys=["image_mr"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image_pet", "image_mr", "label"], source_key="image_pet"),
                ToTensord(keys=["image_pet", "image_mr", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image_pet", "image_mr"]),
                AddChanneld(keys=["image_pet", "image_mr"]),
                Orientationd(keys=["image_pet", "image_mr"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image_pet"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),                
                ScaleIntensityRanged(
                    keys=["image_mr"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image_pet", "image_mr"], source_key="image_pet"),
                ToTensord(keys=["image_pet", "image_mr"]),
            ]
        )

    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms


def infer_post_transforms(args, test_transforms):

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys=["image_pet"],  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys=["image_pet_meta_dict"],  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        ## If monai version <= 0.6.0:
        #AsDiscreted(keys="pred", argmax=True, n_classes=out_classes),
        ## If moani version > 0.6.0:
        AsDiscreted(keys="pred", argmax=True),
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=os.path.join(args.save_path,'seg_result'),
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms



