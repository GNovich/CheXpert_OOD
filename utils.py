from datetime import datetime
import matplotlib.pyplot as plt
import io
import cv2
import torch
import numpy as np
plt.switch_backend('agg')

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def gen_plot_mult(fpr, tpr, labels):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    for i in range(len(labels)):
        plt.plot(fpr[i], tpr[i], linewidth=2, label=labels[i])
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2,
                    (0,255,0),
                    3,
                    cv2.LINE_AA)
    return frame


class StratifiedSampler(torch.utils.data.sampler.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, dataset):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.dataset_len = len(dataset)
        self.numClasses = len(dataset.classes)

        # First, split set by classes
        self.class_indices = []
        for i in range(self.numClasses):
            self.class_indices.append(np.where(dataset.targets == i)[0])
        self.current_class_ind = 0
        self.current_in_class_ind = np.zeros([self.numClasses], dtype=int)

    def gen_sample_array(self):
        # Shuffle all classes first
        for i in range(self.numClasses):
            np.random.shuffle(self.class_indices[i])

        # Construct indset
        indices = np.zeros([self.dataset_len], dtype=np.int32)
        ind = 0
        while(ind < self.dataset_len):
            indices[ind] = self.class_indices[self.current_class_ind][self.current_in_class_ind[self.current_class_ind]]
            # Take care of in-class index
            if self.current_in_class_ind[self.current_class_ind] == len(self.class_indices[self.current_class_ind])-1:
                self.current_in_class_ind[self.current_class_ind] = 0
                # Shuffle
                np.random.shuffle(self.class_indices[self.current_class_ind])
            else:
                self.current_in_class_ind[self.current_class_ind] += 1
            # Take care of overall class ind
            if self.current_class_ind == self.numClasses-1:
                self.current_class_ind = 0
            else:
                self.current_class_ind += 1
            ind += 1
        return indices

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.dataset_len
