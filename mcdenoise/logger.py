# Hendrik Junkawitsch; Saarland University

import os
from datetime import datetime
from loss import *
from scheduler import *
from models.modelchooser import *

class Logger():
    def __init__(self, logdir):
        # Checkpoints for model saves:
        #self.checkpoints = [1, 10, 30, 70, 150, 310, 630]
        self.checkpoints = [(x+1)*10 for x in range(126)]

        print("Created new training logger!")
        logdir = logdir
        now = datetime.now()
        runname = "run_" + str(now.date()) + "|" + datetime.now().strftime("%H:%M:%S")

        self.path = os.path.join(logdir, runname)
        os.mkdir(self.path)

        print(">>>  ", self.path)
        print()

        self.lr_file = open(os.path.join(self.path, "lr_log.txt"), "w")
        self.loss_file = open(os.path.join(self.path, "loss_log.txt"), "w")

        self.meta_file = open(os.path.join(self.path, "meta_log.txt"), "w")

        # list where loss and lr gets temporarily saved
        self.loss = list()
        self.lr = list()

    def log_meta(self, batch_size, lr, num_epochs, loss_id, scheduler_id, in_channels, model_id, optimizer, data):
        
        self.meta_file.write("Training log meta data:\n")
        self.meta_file.write(">>>   batch_size = " + str(batch_size) + "\n")
        self.meta_file.write(">>>   initial_lr = " + str(lr) + "\n")
        self.meta_file.write(">>>   num_epochs = " + str(num_epochs) + "\n")
        self.meta_file.write("Training with " + str(in_channels) + " input channels. \n")

        self.meta_file.write("\n")
        self.meta_file.write("Training data: \n")
        self.meta_file.write(data)
        self.meta_file.write("\n")

        loss = getLossById(loss_id)
        scheduler = getSchedulerById(scheduler_id, optimizer, num_epochs)
        model = get_model(model_id, in_channels)

        self.meta_file.write("\n")
        self.meta_file.write("Optimizer: \n")
        self.meta_file.write(str(optimizer))
        self.meta_file.write("\n")

        self.meta_file.write("\n")
        self.meta_file.write("Loss: \n")
        self.meta_file.write("id = " + str(loss_id) + "\n")
        self.meta_file.write(str(loss))
        self.meta_file.write("\n")
        
        self.meta_file.write("\n")
        self.meta_file.write("Learning rate scheduler: \n")
        self.meta_file.write("id = " + str(scheduler_id) + "\n")
        self.meta_file.write(str(scheduler))
        self.meta_file.write("\n")

        self.meta_file.write("\n")
        self.meta_file.write("Model: \n")
        self.meta_file.write("id = " + str(model_id) + "\n")
        self.meta_file.write(str(model))
        self.meta_file.write("\n")
        
    def log_start(self):
        now = datetime.now()
        self.meta_file.write("Started training: " + str(now.date()) + "|" + datetime.now().strftime("%H:%M:%S"))
        self.meta_file.write("\n")

    def log_end(self, epoch):
        now = datetime.now()
        self.meta_file.write("Finished training: " + str(now.date()) + "|" + datetime.now().strftime("%H:%M:%S"))
        self.meta_file.write("\n")
        self.meta_file.write("Trained for " + str(epoch) + " epochs.")
        self.meta_file.write("\n")

    def log_loss(self, iter, loss):
        self.loss.append((iter, loss))

    def log_lr(self, iter, lr):
        self.lr.append((iter, lr))

    def flush(self):
        for l in self.loss:
            self.loss_file.write(str(l) + "\n")

        for l in self.lr:
            self.lr_file.write(str(l) + "\n")

        self.loss.clear()
        self.lr.clear()

    def close(self):
        self.flush()
        self.loss_file.close()
        self.lr_file.close()
        self.meta_file.close()
        print("Logger closed!")

    def get_run_dir(self):
        return self.path
