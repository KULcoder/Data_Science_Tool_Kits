"""
Write a class: display dynamic train acc and loss on command line.

Design: Should hold a tqdm object, and being able to update it.

How to use:
    1. Create this object with total number of epochs: live_loss = Live_loss(100)
    2. Use it in for loop: for _ in live_loss.epochs:
    3. Update it with acc and loss: live_loss.update(loss, acc)
    4. After epoch epochL live_loss.finish_epoch()

"""
from tqdm import tqdm
import numpy as np

class Live_loss:
    """
    Is designed to replace the epoch iterator.
    """

    def __init__(self, total_epoch=100):
        self.current_epoch = 0
        self.step = 0
        self.epochs = tqdm(range(total_epoch), desc=f"Epoch: {self.current_epoch}")

        self.live_losses = []
        self.live_accs = []

    def update(self, loss, accuracy):

        self.step += 1
        self.live_losses.append(loss)
        self.live_accs.append(accuracy)

        if len(self.live_accs) > 100:
            self.live_accs.pop(0)
        if len(self.live_losses) > 100:
            self.live_losses.pop(0)

        live_loss = np.round(np.mean(self.live_losses), 2)
        live_acc = np.round(np.mean(self.live_accs), 2)
        self.epochs.set_description(
            f"Epoch: {self.current_epoch}, Step: {self.step}, loss: {live_loss:.2f}, acc: {live_acc}"
        )

    def finish_epoch(self):
        self.current_epoch += 1

