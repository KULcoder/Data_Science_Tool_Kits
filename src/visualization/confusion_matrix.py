import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_cm(y_test, y_test_pred, name, labels=None):
    """Plot & display confusion matrix for a given model and dataset.
    TODO: add support for image saving.

    Args:
        y_test (array): Array of true labels.
        y_test_pred (array): Array of predicted labels.
        name (str): Name of the model.
        labels (list, optional): List of labels. Defaults to None.

    Returns:
        None
    """
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm_df, annot=True, cmap=plt.get_cmap('Blues'), fmt="d")
    plt.title("Confusion Matrix of " + name)
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    