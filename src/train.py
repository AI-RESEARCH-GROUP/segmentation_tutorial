# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
if __name__ == '__main__':
    import sys

    sys.path.append("..")

import time
import numpy as np
import tqdm
from multiprocessing import cpu_count
import pathlib  # pathlib是跨平台的、面向对象的路径操作库

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.distributed import DistributedSampler

from src.arg import args
from src.utils.util import proj_root_dir, make_sure_dir
from src.dataset.seg import VocDataset
from src.func.early_stopping import EarlyStopping
from src.func.metric import mean_iou, gen_fusion_matrix
from src.utils.util import print_and_write_log, file_exists
from src.model.my_model import Fcn8s

cuda = torch.cuda.is_available()
if args.gpu < 0:
    cuda = False


def evaluate(model, features, labels, n_class, fusion_matrix):
    model.eval()

    with torch.no_grad():
        loss_func = nn.CrossEntropyLoss(reduction="mean")
        score = model(features)

        # loss
        loss = loss_func(score, labels)

        # mean_iou
        pred_image = score.max(1)[1].cpu().numpy()
        gt_image = labels.cpu().numpy()
        fusion_matrix += gen_fusion_matrix(pred_image, gt_image, n_class)
        
        return loss.item()


def get_loader(split):
    dataloader_config = {'num_workers': 4, 'pin_memory': True}
    # dataloader_config = {'pin_memory': True}
    cropsize = (320, 320)
    ds = VocDataset(split=split, crop_size=cropsize)

    batch_size = 1
    if split == "train":
        batch_size = args.train_batch_size
    elif split == "val":
        batch_size = args.val_batch_size    
    else:
        print_and_write_log("not supported split ")
        exit(-1)
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            drop_last=True,
                            **dataloader_config)

    return ds, loader


def train():
    # ================================================
    # 1) device setting
    # ================================================
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", args.gpu)
        if args.dataparallel:
            print_and_write_log()
            print_and_write_log("Let's use data parallel !!!")
            print_and_write_log()
            # data parallel 自动分配用哪个gpu训练
            device = None
    # ================================================
    # 2) print Data statistics
    # ================================================
    print_and_write_log("""----Data statistics------'
      # num_classes: %s
      # model_name: %s
      --------------------
      """ %
                        ("21", "Fcn-8s",
                         ))
    # ================================================
    # 3) init model/loss/optimizer
    # ================================================
    model = Fcn8s()

    # resume model from pth file
    if args.resume:
        if not file_exists(str(proj_root_dir / 'checkpoints' / 'best_parameters.pth')):
            print_and_write_log("-----------------------------------------")
            print_and_write_log("there is no best_parameters.pth to resume from !!!")
            print_and_write_log("-----------------------------------------")
        else:
            checkpoint = torch.load(str(proj_root_dir / 'checkpoints' / 'best_parameters.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            print_and_write_log("resume model from checkpoints/best_parameters.pth!!!")

    if cuda:
        if args.dataparallel:
            # move to gpu
            model.cuda()

            model = torch.nn.DataParallel(model)

        else:
            model.cuda()

    # ================================================
    # 4) model func init
    # ================================================
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 初始化 early_stopping 对象
    patience = 20  # 当验证集损失在连续10次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True, delta=0.0001, checkpoint_file=str(
        proj_root_dir / "checkpoints/checkpoint.pt"))
    best_mean_iou = 0

    # ================================================
    # 5) train loop
    # ================================================
    train_dataset, train_dataloader = get_loader(split="train")
    val_dataset, val_dataloader = get_loader(split="val")

    def train_one_epoch(epoch_i):
        durations = []
        losses = []
        
        model.train()

        n_class = len(train_dataset.class_names)
        fusion_matrix = np.zeros((n_class,) * 2)      

        print_and_write_log("total train batches: %s" % (len(train_dataloader)))
        for batch_i, (features_batch, labels_batch) in tqdm.tqdm(enumerate(train_dataloader),
                                                                    total=len(train_dataloader),
                                                                    desc='Train epoch=%d' % epoch_i,
                                                                    leave=False):
            t0 = time.time()s

            features_batch = features_batch.float()
            labels_batch = labels_batch.long()            

            if cuda:
                features_batch = features_batch.cuda()
                labels_batch = labels_batch.cuda()

            # forward
            score = model(features_batch)
            loss = loss_func(score, torch.squeeze(labels_batch))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for print
            losses.append(loss.item())
            batch_duration = time.time() - t0
            durations.append(batch_duration)

            # mean_iou            
            pred_image = score.max(1)[1].cpu().numpy()
            gt_image = labels_batch.cpu().numpy()
            fusion_matrix += gen_fusion_matrix(pred_image, gt_image, n_class)
            mean_iou_until_now = mean_iou(fusion_matrix)
            
            print_and_write_log(
                "batch {:05d} | Time(s):{:.2f}s | Loss:{:.4f} | mean_iou:{:.2f}% ".format(batch_i, batch_duration, loss,
                                                                                          mean_iou_until_now * 100))

        mean_iou_epoch = mean_iou(fusion_matrix)       

        return {
            "durations": durations,
            "losses": losses,
            "mean_iou_epoch": mean_iou_epoch,           
        }

    def validate():
        losses = []
        t0 = time.time()
        n_class = len(val_dataset.class_names)
        fusion_matrix = np.zeros((n_class,) * 2)   
       
        print_and_write_log("total val batches: %s" % (len(val_dataloader)))
        for batch_dix, (features_batch, labels_batch) in tqdm.tqdm(
                enumerate(val_dataloader), total=len(val_dataloader),
                desc='validate', leave=False):

            features_batch = features_batch.float()
            labels_batch = labels_batch.long()
            if cuda:
                features_batch = features_batch.cuda()
                labels_batch = labels_batch.cuda()

            loss, num_true, num_gt = evaluate(model, features_batch, labels_batch, n_class, fusion_matrix)
            losses.append(loss)

        average_val_loss = np.mean(losses)
        mean_iou_epoch = mean_iou(fusion_matrix)        

        print_and_write_log(
            "Epoch {:05d} validating: | Time(s):{:.2f}s | Loss:{:.4f} | mean_iou:{:.2f}% ".format(epoch_i, np.mean(
                time.time() - t0), average_val_loss, mean_iou_epoch * 100))       
        
        early_stopping(average_val_loss, model)

        return {
            "average_val_loss": average_val_loss,
            "mean_iou_epoch": mean_iou_epoch,
        }

    print_and_write_log("total epochs: %s" % (args.n_epochs))
    for epoch_i in tqdm.trange(0, args.n_epochs, desc='Train'):
        print_and_write_log()
        print_and_write_log("Epoch {:05d} training...".format(epoch_i))

        one_epoch_result = train_one_epoch(epoch_i)        
        scheduler.step()

        print_and_write_log(
            "Epoch {:05d} training complete...: | Time(s):{:.2f}s | Average Loss:{:.4f} | Average mean_iou:{:.2f}% ".format(
                epoch_i, np.sum(one_epoch_result["durations"]), np.mean(one_epoch_result["losses"]),
                one_epoch_result["mean_iou_epoch"] * 100))        
        
        # ================================================
        # 6) after each epochs ends
        # ================================================
        val_result = validate()
        if val_result["mean_iou_epoch"] > best_mean_iou:
            print_and_write_log(f'Validation mean_iou increased ({val_result["mean_iou_epoch"]*100:.2f}% --> {best_mean_iou*100:.2f}%).  Saving model ...')
            best_mean_iou = val_result["mean_iou_epoch"]
            if args.use_ddp or args.dataparallel:
                torch.save({
                    'epoch': epoch_i,
                    'arch': model.module.__class__.__name__,
                    'optim_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.module.state_dict(),
                    'best_mean_iou': best_mean_iou,
                }, (str(proj_root_dir / 'checkpoints/best_parameters.pth')))
            else:
                torch.save({
                    'epoch': epoch_i,
                    'arch': model.__class__.__name__,
                    'optim_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'best_mean_iou': best_mean_iou,
                }, (str(proj_root_dir / 'checkpoints/best_parameters.pth')))

        if early_stopping.early_stop:
            print_and_write_log("Early stopping")
            # early stop triggered
            break


def main():
    print_and_write_log("[%s] Start training ..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    train()
    print_and_write_log("[%s] End training ..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


if __name__ == '__main__':
    main()
