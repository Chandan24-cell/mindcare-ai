# Model Training

## Dataset

FER2013 Facial Emotion Dataset

Contains thousands of labeled facial expressions across
seven emotion categories.

Classes:

Angry  
Disgust  
Fear  
Happy  
Neutral  
Sad  
Surprise

---

## Model Architecture

Vision Transformer (ViT)

Base model:
vit-base-patch16-224

Advantages:

• Strong image understanding
• Attention-based feature extraction
• High performance on vision tasks

---

## Training Process

Steps:

1 Load dataset
2 Apply preprocessing
3 Initialize pretrained ViT
4 Replace classification head
5 Train on FER2013
6 Save weights

---

## Loss Function

Cross Entropy Loss

---

## Optimizer

AdamW

---

## Training Output

Model checkpoint:

vit_small_emotion.pth