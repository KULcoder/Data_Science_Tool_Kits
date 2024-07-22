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
            self.__train_loader, self.__validation_loader = loaders
            self.__test_loader = None
        elif len(loaders) == 3:
            self.__train_loader, self.__validation_loader, self.__test_loader = loaders

        self.__total_epochs = args['epochs']
        self.__live_loss = Live_loss(self.total_epochs, args['display_interval'])

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # move to the device
        self.__model = self.__model.to(self.__device).float()
        self.__criterion = self.__criterion.to(self.__device).float()

        self.__train_accs = []
        self.__train_losses = []
        self.__val_accs = []
        self.__val_losses = []
        self.__best_val_acc = 0.0
        self.__best_val_acc_epoch = None

    def train(self):
        start_time = time.time()

        for epoch in self.__live_loss.epochs:
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.test(validate=True)
        
            if val_acc > self.__best_val_acc:
                self.__best_val_acc = val_acc
                self.__best_val_acc_epoch = epoch

            if self.__lr_schduler is not None:
                self.__lr_schduler.step()
                # Some lr_schuler might requires the input of val_loss

        end_time = time.time()
        run_time = round(end_time - start_time, 2)

        return self.__best_val_acc, run_time

    def test(self, validate=True):
        self.__model.eval()
        if not validate or not self.__test_loader:
            loader = self.__validation_loader
        else:
            loader = self.__test_loader

        test_loss = 0.0
        test_correct = 0.0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.__device), labels.to(self.__device)

                logits = self.__model(images)
                loss, correct = self.__compute_loss_accuracy(logits, labels)
                test_loss += loss.detach().cpu()
                test_correct += correct

        final_loss = test_loss.detach().cpu().item() / len(loader)
        final_acc = test_correct * 100 / len(loader.dataset)

        if validate:
            self.__val_accs.append(final_acc)
            self.__val_losses.append(final_loss)

        return final_loss, final_acc

    def train_one_epoch(self):
        self.__model.train()
        train_loss = 0.0
        train_correct = 0.0

        for images, labels in self.__train_loader:
            images, labels = images.to(self.__device), labels.to(self.__device)

            logits = self.__model(images)
            loss, correct = self.__compute_loss_accuracy(logits, labels)
            train_loss += loss.detach().cpu()
            train_correct += correct

            self.__live_loss.update(loss.detach().cpu(), correct / labels.shape[0])

            # back prop
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

        self.__train_accs.append(train_correct * 100 / total_number_images)
        self.__train_losses.append(train_loss.detach().cpu().item() / total_number_batches)
        self.__live_loss.finish_epoch()

        return self.__train_losses[-1], self.__train_accs[-1]

    def __compute_loss_accuracy(self, logits, labels):
        loss = self.__criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        return loss, correct

    def get_model(self):
        return self.__model


