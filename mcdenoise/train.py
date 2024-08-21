# Hendrik Junkawitsch; Saarland University

# This is the main training module.
# Executing train.py will start the training process.

from torch.serialization import save
from dataset import *
from config import Config
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from logger import Logger
from scheduler import Schedule

def train(save_path):
    # Configuration module
    conf = Config()
    
    data = "training_data/data"

    # Training Logger
    logger = Logger("runs")

    # Training data set
    dataset     = DataSet(conf.in_channels, data)
    dataloader  = DataLoader(dataset=dataset, batch_size=conf.batchsize, shuffle=True, num_workers=15)

    # Training device (Standard GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on: ", device)

    # Model
    model = conf.model
    model.to(device)

    # Criterion
    criterion = conf.loss
    criterion.to(device)

    # Optimizer and scheduler
    optimizer = conf.optimizer
    if conf.scheduler_id != Schedule.CONST: scheduler = conf.scheduler

    logger.log_meta(conf.batchsize, conf.lr, conf.num_epochs, conf.loss_id, conf.scheduler_id, conf.in_channels, conf.model_id, conf.optimizer, data)

    # Training loop with Keyboard interrupt
    try: 
        logger.log_start()
        iter = 0
        for epoch in range(conf.num_epochs):
            i = 0
            logger.log_lr(epoch, optimizer.param_groups[0]['lr'])
            if epoch in logger.checkpoints:
                torch.save(model, os.path.join(logger.get_run_dir(), "checkpoint"+str(epoch)+".pt"))
                print()
                print(">>>  Saved a checkpoint ! epoch", epoch)
                print()
            for data in dataloader:
                # Moving ground truth and train tensor to device
                (gt, t) = data[0].to(device), data[1].to(device)
                recon = model(t)
                loss = criterion(recon, gt)
                print("epoch [", epoch,"/", conf.num_epochs,"] ; batch_num ", i, " ; loss ", loss.item(), " ; lr ", optimizer.param_groups[0]['lr'])
                logger.log_loss(iter, loss.item())
                del(recon)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del(loss)
                i += 1
                iter += 1
            logger.flush()
            if conf.scheduler_id != Schedule.CONST: 
                scheduler.step()
            print(" finished!")
    except KeyboardInterrupt:
        print(" Keyboard interrupt. Finished training")
        logger.log_end(epoch)
        logger.close()

        # Saving model
        torch.save(model, os.path.join(logger.get_run_dir(), save_path))
        print("Model saved!")
        return

    print("Finished training")
    logger.log_end(epoch)
    logger.close()

    # Saving model
    torch.save(model, os.path.join(logger.get_run_dir(), save_path))
    print("Model saved!")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == "__main__":
    print("Starting training...")
    save_path = "model.pt"
    train(save_path)