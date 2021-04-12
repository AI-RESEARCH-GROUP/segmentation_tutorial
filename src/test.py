# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if __name__ == '__main__':
    import sys

    sys.path.append("..")

import time
import numpy as np
import tqdm
from PIL import Image
import os.path as path
import os
import pathlib

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from src.arg import args
from src.dataset.seg import SegDataset
from src.func.metric import mean_iou, gen_fusion_matrix, compute_class_true
from src.utils.util import print_and_write_log, file_exists, proj_root_dir, make_sure_dir
from src.model.my_model import MyModel


def get_test_loader():
    dataloader_config = {'num_workers': 4, 'pin_memory': True}
    ds = SegDataset(split="test", no_crop=True)
    batch_size = args.test_batch_size

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **dataloader_config)
    return ds, loader


test_dataset, test_dataloader = get_test_loader()


def get_labels():
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128],[128,128,128],[255,0,0]])


def single_label_to_pil_image(label_mask):
    n_classes = 9
    label_colours = get_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    pil_img = Image.fromarray(np.uint8(rgb))
    return pil_img


def relative_path( img_path):
    return str(pathlib.Path(img_path).relative_to(args.raw_data_dir))

#后处理，保证只分割出一种缺陷
def post_process(pred):
    mask = np.unique(pred)
    tmp = [1]
    for v in mask:
        if v != 0:
            tmp.append(np.sum(pred==v))    
    max_v = mask[np.argmax(tmp)]
    for v in mask:
        if v != 0 and v != max_v:
            pred[pred==v] = max_v
    return pred

def gen_img(features, pred_box_batch, ii_batch):
    for i in range(features.shape[0]):
        img_path = test_dataset.total_path[ii_batch[i]]["img_path"]
        print_and_write_log("handing %s ..." % (relative_path(img_path)))
        base_name = path.basename(str(img_path))
        base_name = os.path.splitext(base_name)[0]

        #后处理
        pred = post_process(pred_box_batch[i])
        pred_img = single_label_to_pil_image(pred)
        original_img = Image.open(img_path)

        if pred_img.size != original_img.size and original_img.size[0] < original_img.size[1]:
            # 1024*1280 变成 1280*1024,再变回来
            pred_img = pred_img.rotate(-90, expand=True)
        merged_img = Image.blend(original_img, pred_img, 0.7)

        # save
        make_sure_dir(proj_root_dir / 'output/img/')
        original_img.save(str(proj_root_dir / 'output/img/' / ('%s.png' % (base_name,))))
        merged_img.save(str(proj_root_dir / 'output/img/' / ('%s_result.png' % (base_name,))))

        pred_img.close()
        original_img.close()
        merged_img.close()


def evaluate(model, features, labels, n_class, ii_batch, fusion_matrix, num_true, num_gt):
    model.eval()

    with torch.no_grad():
        loss_func = nn.CrossEntropyLoss(reduction="mean")
        score = model(features)

        # loss
        loss = loss_func(score, labels)

        #mean_iou
        pred_image = score.max(1)[1].cpu().numpy()
        gt_image = labels.cpu().numpy()        
        fusion_matrix += gen_fusion_matrix(pred_image, gt_image, n_class)

        # 统计预测正确的数量
        num_true, num_gt = compute_class_true(pred_image, gt_image, num_true, num_gt)
        
        pred_box_batch = np.array(pred_image, dtype=np.uint8)
        gen_img(features, pred_box_batch, ii_batch)

        return loss.item(), num_true, num_gt


def test():
    if not file_exists(str(proj_root_dir / 'checkpoints/best_parameters.pth')):
        print_and_write_log()
        print_and_write_log("please run train.py first !!!")
        print_and_write_log()
        exit(-1)

    # ================================================
    # 1) use cuda
    # ================================================
    cuda = torch.cuda.is_available()
    if args.gpu < 0:
        cuda = False
    # ================================================
    # 1) device setting
    # ================================================
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", args.gpu)
        if args.use_ddp:
            print_and_write_log()
            print_and_write_log("Let's use" + str(torch.cuda.device_count()) + "GPUs !!!")
            print_and_write_log()
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        elif args.dataparallel:
            print_and_write_log()
            print_and_write_log("Let's use data parallel !!!")
            print_and_write_log()
            # data parallel 自动分配用哪个gpu训练
            device = None

    print_and_write_log("""----Data statistics------'
      # num_classes: %s
      # model_name: %s
      --------------------
      """ %
                        ("9", "MyModel",
                         ))
    
    model = MyModel(channels=3, class_nums=9)
    checkpoint = torch.load(str(proj_root_dir / 'checkpoints/best_parameters.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print_and_write_log("resume model from %s !!!" % (str(proj_root_dir / 'checkpoints/best_parameters.pth')))

    if cuda:
        if args.use_ddp:
            # move to gpu
            model.cuda()

            # ddp model
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[local_rank],
                                                              output_device=local_rank)
            
        elif args.dataparallel:
            # move to gpu
            model.cuda()

            model = torch.nn.DataParallel(model)
            
        else:
            model = model.to(device)    

    # ================================================
    # 4) eval/test
    # ================================================
    def test_():
        losses = []
        mean_iou_list = []
        t0 = time.time()
        n_class = len(test_dataset.class_names)

        num_true = np.zeros(n_class-1)
        num_gt = np.zeros(n_class-1)

        fusion_matrix = np.zeros((n_class,)*2)
        print_and_write_log("total test batches: %s" % (len(test_dataloader)))
        for batch_idx, (features_batch, labels_batch, ii_batch) in tqdm.tqdm(
                enumerate(test_dataloader), total=len(test_dataloader),
                desc='test', leave=False):

            # img_path = test_dataset.total_path[ii_batch[ii_batch]]["img_path"]
            # print(img_path)

            features_batch = features_batch.float()
            labels_batch = labels_batch.long()
            if cuda:
                features_batch = features_batch.cuda()
                labels_batch = labels_batch.cuda()
            
            loss, num_true, num_gt = evaluate(model, features_batch, labels_batch, n_class, ii_batch, fusion_matrix, num_true, num_gt)                        
            losses.append(loss)

        average_test_loss = np.mean(losses)
        mean_iou_epoch = mean_iou(fusion_matrix)
        accuracy_test = num_true/num_gt        
        
        time_span = time.time() - t0
        time_per_img = time_span / len(test_dataloader.dataset)

        print_and_write_log(
            "Test : | Time(s):{:.2f}s | Time-per-img(ms):{:.3f}ms | Loss:{:.4f} | mean_iou {:.2f}% "
                .format(time_span, time_per_img * 1000, average_test_loss, mean_iou_epoch * 100))

        print_and_write_log(f'test_accuracy--HSPSC:{accuracy_test[0]*100:.2f}% | HSPSD:{accuracy_test[1]*100:.2f}% | HSWZC:{accuracy_test[2]*100:.2f}% | HSWZD:{accuracy_test[3]*100:.2f}% | HSQBC:{accuracy_test[4]*100:.2f}% | HSQBD:{accuracy_test[5]*100:.2f}% | HSCSC:{accuracy_test[6]*100:.2f}% | HSCSD:{accuracy_test[7]*100:.2f}%')

    test_()


def main():
    print_and_write_log("[%s] Start test ..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    test()
    print_and_write_log("[%s] End test ..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


if __name__ == '__main__':
    main()
