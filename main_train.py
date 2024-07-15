
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
#from attention import SA

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms


import os
import numpy as np
from tqdm import tqdm
import multiprocessing
import argparse


parser = argparse.ArgumentParser(description='3D UX-Net hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root_pet', type=str, default='./data_PET', required=False, help='Root folder of all your images and labels')
parser.add_argument('--root_mr', type=str, default='./data_MR', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='pet_mr', required=False, help='Datasets: {PET, MR, PET_MR}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='3DUXNET', help='Network models: {nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=20000, help='Maximum iteration steps for training') #20000
parser.add_argument('--eval_step', type=int, default=80, help='Per steps to perform validation') #80
parser.add_argument('--out_classes', type=int, default=45, help='out lasses of brain dataset segmentation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Used GPU: {}'.format(args.gpu))

    train_samples, valid_samples = data_loader(args)

    if args.dataset == 'pet_mr':
        train_files = [
            {"image_pet": image_pet_name, "image_mr": image_mr_name, "label": label_name}
            for image_pet_name, image_mr_name, label_name in zip(train_samples['images_pet'], train_samples['images_mr'], train_samples['labels'])
        ]

        val_files = [
            {"image_pet": image_pet_name, "image_mr": image_mr_name, "label": label_name}
            for image_pet_name, image_mr_name, label_name in zip(valid_samples['images_pet'], valid_samples['images_mr'], valid_samples['labels'])
        ]
    else:
        train_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
        ]

        val_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
        ]


    set_determinism(seed=0)

    train_transforms, val_transforms = data_transforms(args)

    ## Train Pytorch Data Loader and Caching
    print('Start caching datasets!')
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=args.cache_rate, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    ## Valid Pytorch Data Loader and Caching
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)


    ## Load Networks
    device = torch.device("cuda:0")
    if args.network == '3DUXNET':
        model = UXNET(
            in_chans=1,
            out_chans=args.out_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        ).to(device)       
    elif args.network == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=args.out_classes,
            feature_size=48,
            use_checkpoint=False,
        ).to(device)
    elif args.network == 'UNETR':
        model = UNETR(
            in_channels=1,
            out_channels=args.out_classes,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)

    print('Chosen Network Architecture: {}'.format(args.network))

    if args.pretrain == True:
        print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
        model.load_state_dict(torch.load(args.pretrained_weights))

    ## Define Loss function and optimizer
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    print('Loss for training: {}'.format('DiceCELoss'))
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))


    root_dir = os.path.join(args.output)
    if os.path.exists(root_dir) == False:
        os.makedirs(root_dir)
        
    t_dir = os.path.join(root_dir, 'tensorboard')
    if os.path.exists(t_dir) == False:
        os.makedirs(t_dir)
    writer = SummaryWriter(log_dir=t_dir)

    def validation(epoch_iterator_val):
        model.eval()
        dice_vals = list()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_pet_inputs, val_mr_inputs, val_labels = (batch["image_pet"].cuda(), batch["image_mr"].cuda(), batch["label"].cuda())
                combined_inputs = torch.cat((val_pet_inputs, val_mr_inputs), dim=1)
                val_outputs = sliding_window_inference(combined_inputs, (96, 96, 96), 2, model)            
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 80.0, dice)
                )
            dice_metric.reset()
        mean_dice_val = np.mean(dice_vals)
        writer.add_scalar('Validation Segmentation Dice', mean_dice_val, global_step)
        return mean_dice_val


    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x1, x2, y = (batch["image_pet"].cuda(), batch["image_mr"].cuda(), batch["label"].cuda())
            x_in = torch.cat((x1, x2), dim=1)
            logit_map = model(x_in)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            if (
                global_step % (2*eval_num) == 0 and global_step != 0
            ) or global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
            writer.add_scalar('Training Segmentation Loss', loss.data, global_step)
            global_step += 1
        return global_step, dice_val_best, global_step_best


    max_iterations = args.max_iter
    print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
    eval_num = args.eval_step
    post_label = AsDiscrete(to_onehot=args.out_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_classes)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )      

if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    main()



