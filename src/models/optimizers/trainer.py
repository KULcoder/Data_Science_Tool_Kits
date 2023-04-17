import numpy as np
import copy
from src.data.batch_generator import batch_generator
from src.data.shuffle_data import shuffle_data
from src.metrics.visualization.live_plot import live_plot

class Trainer:
    """
    Framework for gradient methods training process.
    When use this method for complex model and early stop method, pay attention to the double descent problem.
    Early stop is implemented in the way that requires consistent worse performance in the validation loss to stop the training.
    """

    def __init__(self):
        pass

    def train(self, optimier, model, train_data, validation_data, batch_size, learning_rate, epochs, patience, early_stop = False, is_classification=True):
        """
        Training procedure.
        """
        X_train, y_train = train_data
        X_validation, y_validation = validation_data
        model.set_optimizer(optimier)

        # records train and validation information
        if is_classification:
            logs = {'train': {'loss': [], 'accuracy': []}, 'validation': {'loss': [], 'accuracy': []}}
        else:
            # regression, we only use loss to measure the performance
            logs = {'train': {'loss': []}, 'validation': {'loss': []}}

        if early_stop:
            # early stop related
            patience_count = patience
            best_loss = np.inf
            best_epoch = 0
            best_model = None

        for epoch in range(epochs): # total number of epochs

            X, y = shuffle_data(X_train, y_train)

            if is_classification:
                batch_log = {'loss': [], 'accuracy': []}
            else:
                batch_log = {'loss': []}

            for X_batch, y_batch in batch_generator(X, y, batch_size): # each batch 
                predictions = model.forward(X_batch, y_batch)
                loss = model.loss(predictions, y_batch)
                model.update(loss, learning_rate) # itself calculate the gradient

                if is_classification:
                    accuracy = model.accuracy(predictions, y_batch)
                    batch_log['accuracy'].append(accuracy)
                batch_log['loss'].append(loss)

            # calculate the average loss and accuracy for each epoch
            if is_classification:
                logs['train']['accuracy'].append(np.mean(batch_log['accuracy']))
            logs['train']['loss'].append(np.mean(batch_log['loss']))

            # calculate the validation loss and accuracy for each epoch
            predictions = model.forward(X_validation, y_validation)
            loss = model.loss(predictions, y_validation)

            if is_classification:
                y_pred = predictions.argmax(axis=1)
                acc = accuracy(y_pred, y_validation)
                logs['validation']['accuracy'].append(acc)
            logs['validation']['loss'].append(loss)

            # early stop
            if early_stop:
                if loss < best_loss:
                    patience_count = patience
                    best_loss = loss
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                else:
                    if patience_count == 0:
                        print("Early stop at epoch: {}".format(epoch))
                        break
                    else:
                        patience_count -= 1

            # plot the training process
            live_plot(logs)

        if early_stop:
            return best_model, best_epoch, best_loss, logs
        else:
            return model, logs
    





