import matplotlib.pyplot as plt
from IPython.display import clear_output

def live_plot(logs, title="Training Progress"):
    """
    Create live plots for gradient methods. This is specifically designed for jupyter notebook.
    """
    clear_output(wait=True)
    plt.figure(figsize=(20, 5))

    if 'accuracy' in logs['train']:
        plt.subplot(1, 2, 1)
        plt.title('Accuracy')
        plt.plot(logs['train']['accuracy'], label='train')
        plt.plot(logs['validation']['accuracy'], label='validation')
        plt.legend(loc='best')

        plt.subplot(1, 2, 2)
        plt.title('Loss')
        plt.plot(logs['train']['loss'], label='train')
        plt.plot(logs['validation']['loss'], label='validation')
        plt.legend(loc='best')
        

    else:
        plt.title('Loss')
        plt.plot(logs['train']['loss'], label='train')
        plt.plot(logs['validation']['loss'], label='validation')
        plt.legend(loc='best')

    plt.suptitle(title)
    plt.show()

