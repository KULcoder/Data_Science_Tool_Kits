"""
A class: display dynamic train acc and loss on command line.
Useful for training deep learning models, and give you a sense about your training
while not using big monitor libs like tensorboard.

requires: tqdm, numpy, pytorch

Question:
1. Do we want to add something to display live plot? 
    Seems too complex and unnecessary
2. Do we want to display something about validation loss?
    Mechanism like early-stopping should be implemented outside this class, not
    important to have


How to use:
    1. Create this object with total number of epochs: live_loss = Live_loss(100)
    2. Use it in for loop: for _ in live_loss.epochs:
    3. Update it with acc and loss: live_loss.update(loss, acc)
    4. After epoch epochL live_loss.finish_epoch()

"""
from tqdm import tqdm
import numpy as np
import torch

class Live_loss:
    """
    Is designed to replace the epoch iterator.
    """

    def __init__(self, total_epoch=100, display_interval=1, gpu_usage=False):
        self.current_epoch = 1
        self.step = 0
        self.epochs = tqdm(range(total_epoch))
        self.display_interval = display_interval
        self.gpu_usage = gpu_usage

        self.live_losses = []
        self.live_accs = []

    def update(self, loss, accuracy):

        self.step += 1
        self.live_losses.append(loss)
        self.live_accs.append(accuracy)

        self.live_losses = self.live_losses[-100:]
        self.live_accs = self.live_accs[-100:]


        if self.display_interval and self.step % self.display_interval == 0:
            live_loss = np.round(np.mean(self.live_losses), 2)
            live_acc = np.round(np.mean(self.live_accs) * 100, 2)
            description = f"Step: {self.step}, loss: {live_loss:.2f}, acc: {live_acc}%"

            if self.gpu_usage and torch.cuda.is_available:
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
                gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
                gpu_memory_percentage = gpu_memory_reserved / gpu_memory_total
                description += f", gpu_mem: {gpu_memory_percentage:.2f}%"

            self.epochs.set_description(description)

    def finish_epoch(self):
        self.current_epoch += 1

