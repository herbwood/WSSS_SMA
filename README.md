# Learning to Detour: Shortcut Mitigating Augmentation for Weakly Supervised Semantic Segmentation

Official PyTorch implementation of  **“Learning to Detour: Shortcut Mitigating Augmentation for Weakly Supervised Semantic Segmentation”**  
(arXiv: [[2405.18148]((https://arxiv.org/pdf/2405.18148))](https://arxiv.org/pdf/2405.18148))

---

## 🧠 Abstract

Weakly supervised semantic segmentation (WSSS) employing weak forms of labels has been actively studied to alleviate the annotation cost of acquiring pixel-level labels. However, classifiers trained on biased datasets tend to exploit shortcut features and make predictions based on spurious correlations between certain backgrounds and objects, leading to a poor generalization performance. In this paper, we propose shortcut mitigating augmentation (SMA) for WSSS, which generates synthetic representations of object-background combinations not seen in the training data to reduce the use of shortcut features. Our approach disentangles the object-relevant and background features. We then shuffle and combine the disentangled representations to create synthetic features of diverse object-background combinations. SMA-trained classifier depends less on contexts and focuses more on the target object when making predictions. In addition, we analyzed the behavior of the classifier on shortcut usage after applying our augmentation using an attribution method-based metric. The proposed method achieved the improved performance of semantic segmentation result on PASCAL VOC 2012 and MS COCO 2014 datasets. 

---

## ⚙️ Installation

```bash
git clone https://github.com/herbwood/WSSS_SMA.git
cd WSSS_SMA
conda create -n sma python=3.8
conda activate sma
pip install -r requirements.txt
```

## 🚀 Training

### Step 1: Edit ```run.sh```
Update your save path before training:
```bash
SAVEPATH=/PATH/TO/SAVE/WEIGHTS
```

### Step 2: Run SMA training
Run the training script:
```bash
bash run.sh
```
