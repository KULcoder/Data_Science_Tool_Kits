"""
Design: Write a class of trainer.

This trainer should hold all of the objects required for training a pytorch model.

"""




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.live_loss import Live_loss
import torch
import time

class Trainer(object):

    def __init__(self, args, model, optimizer, lr_scheduler, criterion, loaders):

        self.__model = model
        self.__optimizer = optimizer
        self.__lr_schduler = lr_scheduler
        self.__criterion = criterion

        if len(loaders) == 2:
            self.__train_loader, self.__test_loader = loaders
            self.__validation_loader = None
        elif len(loaders) == 3:
            self.__train_loader, self.__validation_loader, self.__test_loader = loaders

        self.__total_epochs = args['epochs']
        self.__live_loss = Live_loss(self.total_epochs)

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # move to the device
        self.__model = self.__model.to(self.__device).float()
        self.__criterion = self.__criterion.to(self.__device).float()

    def train_one_epoch(self):
        self.__model.train()
        for images, labels in self.__train_loader:
            images, labels = images.to(self.__device), labels.to(self.__device)

            logits = self.__model(images)
            loss, correct = self.__compute_loss_accuracy(logits, labels)

    def train():
        pass

    def __compute_loss_accuracy(self, logits, labels):
        loss = self.__criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        return loss, correct

    def get_model(self):
        return self.__model


