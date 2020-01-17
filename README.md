# Emotion Detection from Hindi Texts Using ULMFiT

In this project I will showcase a ULMFiT model and use it for Emotion Detection. ULMFiT is the technique of using transfer learning for text classification task.

# Transfer Learning

Transfer learning is the technique of using weights from a pre-trained deep neural network and tweaking them a bit to suit our application. In other words, it is applying the knowledge of an already trained model to a different but related problem.
It is suited to applications having a small dataset and also reduces computation time.

# What is ULMFit

ULMFiT stands for Universal Language Model Fine-tuning for Text Classification, a technique introduced by Jeremy Howardand Sebastian Ruder. It is a technique to incorporate transfer learningin NLP tasks.
  - https://en.wikipedia.org/wiki/Transfer_learning
  - https://arxiv.org/abs/1801.06146
  
USPs of ULMFiT is-
  - Discriminative fine-tuning
  - Slanted triangular learning rates
  - Gradual unfreezing
   
# Dataset
The dataset is created manually as there’s no pre-existing dataset for Hindi Emotion Detection. It comprises of 5 labels: Angry, Happy, Neutral, Sad and Excited.

Each entry of the dataset is then converted to a text file which is stored in a folder of the class to which it belongs.

```
CLASSES = ['angry','excited','happy','neutral','sad']
```
```
def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)
```
```
trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')
```

The dataset consists of 5 classes- Angry, Excited, Happy, Neutraland Sad.

The get_texts() function loads the data and stores all the texts in trn_textsand val_textsand their respective labels in trn_labelsand val_labels.
