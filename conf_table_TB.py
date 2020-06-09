from textwrap import wrap
import re
import itertools
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix',
                          tensor_name = 'MyFigure/image', normalize=True):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(4,4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges', vmin=0.5, vmax=1)

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=14)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], '.2f') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    return fig


def plot_auc_vector(values, labels):
    auc_mat = np.array([values])

    np.set_printoptions(precision=4)
    fig = matplotlib.figure.Figure(figsize=(4,4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(auc_mat, cmap='Oranges', vmin=0.5, vmax=1)

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('class', fontsize=10)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('AUC', fontsize=10)
    ax.set_yticks([])
    ax.set_yticklabels([])

    for i, j in itertools.product(range(auc_mat.shape[0]), range(auc_mat.shape[1])):
        ax.text(j, i, format(auc_mat[i, j], '.2f') if auc_mat[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    return fig