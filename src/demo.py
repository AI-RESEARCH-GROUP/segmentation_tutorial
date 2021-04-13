if __name__ == '__main__':
    import sys

    sys.path.append("..")


import torch
import os
import time
import random
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from src.model.my_model import Fcn8s
from src.utils.util import proj_root_dir, make_sure_dir

cuda = torch.cuda.is_available()

use_torchvision = True

if use_torchvision:
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
else:
    #加载模型
    checkpoint = torch.load(str(proj_root_dir / 'checkpoints/best_parameters.pth'))
    model = Fcn8s(n_class=21)    
    model.load_state_dict(checkpoint['model_state_dict'])

if cuda:
    model.cuda()

#读取图像
def read_img(image_path):
    res= os.walk(image_path)
    images = []    
    for root,dirs,files in res:
        for f in files:
            images.append(os.path.join(root,f))
    return images

def process(img_path):
    img = Image.open(img_path)
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.normalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    c,h,w = img_tensor.size()    
    img_tensor = img_tensor.reshape(1,c,h,w)
    return img, img_tensor


#做预测
def predict(images):
    model.eval()
    image = []
    predict = []
    for img_path in images:
        img, img_tensor = process(img_path)
        image.append(img)        
        img_tensor = img_tensor.cuda()
        pred = model(img_tensor)['out']
        _,c,h,w = pred.size()
        pred = pred.reshape(c,h,w)  
        pred_image = pred.max(0)[1].cpu().numpy()        
        predict.append(pred_image)
    result = list(zip(image, predict, images))
    return result

#生成图像
def pred_to_image(label_mask):
    n_classes = 21
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

def get_labels():    
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

#保存结果
def gen_img(result):
    for img, pred, path in result:
        print("handing %s ..." % (path))
        base_name = os.path.split(path)[1]
        base_name = os.path.splitext(base_name)[0]
        pred_img = pred_to_image(pred)
        merged_img = Image.blend(img, pred_img, 0.7)
        make_sure_dir(proj_root_dir / 'result/')        
        merged_img.save(str(proj_root_dir / 'result' / ('%s_result.png' % (base_name))))

def demo():
    image_path = os.path.join(proj_root_dir, 'pic')
    images = read_img(image_path)
    result = predict(images)
    gen_img(result)

def main():
    print("[%s] Start demo ..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    demo()
    print("[%s] End demo ..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

if __name__ == '__main__':
    main()



    


