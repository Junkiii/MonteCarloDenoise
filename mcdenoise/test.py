# Hendrik Junkawitsch; Saarland University

# This is the testing module. 
# Executing test.py will denoise an image from a dataset

from torch import tensor
from dataset import *
from models.modelchooser import *
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from config import Aux
from image_similarity_measures.quality_metrics import rmse, ssim, fsim, issm, sre, sam, uiq
#from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import torch.nn as nn
import time

def load_model(path):
    return torch.load(path)

def show_result(res):
    plt.imshow(res)
    plt.show()

def save_image(name, img, div):
    if div: img = img * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)

def get_sample(idx, data):
    gt, t = data.__getitem__(idx)
    gt = gt.unsqueeze(0)
    t = t.unsqueeze(0)
    return gt, t

def denoise(model, t):
    res = model(t)
    res = res.detach()
    res = res[0].permute(1,2,0)
    res = np.array(res)
    return res

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == "__main__":
    # Arguments:
    model_path      = "runs/cap_eva/run_2021-11-20|10:25:50/checkpoint230.pt"
    data_path       = "validation_data/data_small"
    num_features    = Aux.NDSN
    idx             = 23
    # ----------
    
    model = load_model(model_path)
    model.to("cpu")
    
    dataset = DataSet(num_features, data_path, crop=False)
    gt, t = get_sample(idx, dataset)
    (noisy, albedo, normal, diffuse, specular, gt) = dataset.load_sample(idx)

    # Generating denoised image with the trained network
    start = time.time()
    res = denoise(model, t)
    end = time.time()
    print(end-start)
    res_comp = res * 255;

    #print(mse(res_comp,gt))
    #print(ssim(res_comp,gt))

    save_image("out/gt.png",        gt,       False)
    save_image("out/noisy.png",     noisy,    False)
    save_image("out/albedo.png",    albedo,   False)
    save_image("out/normal.png",    normal,   False)
    save_image("out/diffuse.png",   diffuse,  False)
    save_image("out/specular.png",  specular, False)
    save_image("out/denoised.png",  res,      True )

