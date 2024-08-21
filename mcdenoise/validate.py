# Hendrik Junkawitsch; Saarland University

# Validation module documenting the loss

from dataset import *
from loss import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
import image_similarity_measures as ism
from image_similarity_measures.quality_metrics import rmse, ssim, fsim, issm, sre, sam, uiq
#from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

def load_model(path):
    return torch.load(path)

def get_sample(idx, data):
    gt, t = data.__getitem__(idx)
    gt = gt.unsqueeze(0)
    t = t.unsqueeze(0)
    return gt, t

def denoise(model, t, device):
    t = t.to(device)
    res = model(t)
    res = res.to("cpu")
    res = res.detach()
    res = res[0].permute(1,2,0)
    res = np.array(res)
    return res

def avg(l):
    return sum(l) / len(l)

def write_log(path, l):
    file = open(path, "w")
    file.write("Average\n")
    file.write(str(avg(l)))
    file.write("\n\n")
    for i in l:
        file.write(str(i) + "\n")
    file.close()

def validate(run, model_name, data, name):
    model   = load_model(os.path.join(run, model_name))
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = DataSet(Aux.NDSN, data, crop=False)
    size = dataset.__len__()

    # Loss logs for the validation set
    #mae_log = list()
    #mse_log = list()
    ssim_log = list()
    #msssim_log = list()
    #rmse_log = list()
    #sre_log = list()
    #fsim_log = list()
    #issm_log = list()
    #sam_log = list()
    #uiq_log = list() 

    # Iterate over every sample in the validation set and 
    # calculate the various losses.
    for i in range(size):
        print(i)
        gt, t = get_sample(i, dataset)
        # (noisy, albedo, normal, gt) = dataset.load_sample(i)
        (noisy, albedo, normal, diffuse, specular, gt) = dataset.load_sample(i)
        res = denoise(model, t, device)

        #res = torch.tensor(res)
        #gt = torch.tensor(gt)
        
        #mae_log.append(np.array(mae(res, gt)))
        
        res = np.array(res) * 255
        gt = np.array(gt)

        #mse_log.append(mse(res, gt))
        ssim_log.append(ssim(res, gt))
        #msssim_log.append(msssim(res,gt))
        #rmse_log.append(rmse(res, gt))
        #sre_log.append(sre(res, gt))
        #fsim_log.append(fsim(res, gt))
        #issm_log.append(issm(res, gt))
        #sam_log.append(sam(res, gt))
        #uiq_log.append(uiq(res, gt))
    
    #print("avg(MAE) ="  , avg(mae_log))
    #print("avg(MSE) ="  , avg(mse_log))
    print("avg(SSIM) =" , avg(ssim_log))
    #print("avg(MSSSIM) =" , avg(msssim_log))
    #print("avg(RMSE) =" , avg(rmse_log))
    #print("avg(SRE) ="  , avg(sre_log))
    #print("avg(FSIM) =", avg(fsim_log))
    #print("avg(ISSM) =", avg(issm_log))
    #print("avg(SAM) =" , avg(sam_log))
    #print("avg(UIQ) =" , avg(uiq_log))
    
    folder_name = "validation_" + name + "_" + model_name
    folder_path = os.path.join(run, folder_name)
    try:
        os.mkdir(folder_path)    
    except: pass

    #write_log(os.path.join(folder_path, "mae_log")  , mae_log)
    #write_log(os.path.join(folder_path, "mse_log")  , mse_log)
    write_log(os.path.join(folder_path, "ssim_log") , ssim_log)
    #write_log(os.path.join(folder_path, "msssim_log") , msssim_log)
    #write_log(os.path.join(folder_path, "rmse_log") , rmse_log)
    #write_log(os.path.join(folder_path, "sre_log")  , sre_log)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == "__main__":
    run = "runs/cap_eva/run_2021-11-26|04:52:46"
    model_names = list()
    checkpoints = [(x+1)*10 for x in range(126)]
    for c in checkpoints:
        model_names.append("checkpoint" + str(c) + ".pt")

    model_names.append("model.pt")

    for model_name in model_names:
        print(model_name)
        validate(run, model_name, "validation_data/data_small", "valid")
        validate(run, model_name, "training_data/data_small", "train")