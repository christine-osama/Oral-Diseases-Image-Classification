# Oral Disease Image Classification

A deep learning pipeline that classifies oral disease images into six categories: Calculus, Caries, Gingivitis, Hypodontia, Tooth Discoloration, and Ulcers. Built and trained end-to-end in Google Colab, with an interactive Gradio interface for deployment.

## Overview

This project compares a convolutional network trained from scratch against three ImageNet pretrained backbones adapted through transfer learning, tunes the strongest backbone's hyperparameters, and deploys the final model for interactive use.

The goal was not just to train one model, but to build a full comparative pipeline: baseline, transfer learning, tuning, evaluation, and deployment, with each stage justified by the results of the one before it.

## Dataset

[Oral Diseases](https://www.kaggle.com/datasets/salmansajid05/oral-diseases) on Kaggle, downloaded via `kagglehub`.

The raw dataset includes a folder built for a YOLO object detection task that bundles four diseases together for bounding box annotation rather than classification. This folder is identified and excluded during preprocessing so it is not mistaken for a seventh, incorrect class. The remaining six classes are collected from their respective folders (including both augmented and original image sets, where the dataset provides both) and split into train, validation, and test sets.

## Approach

**Baseline**
A custom CNN with four convolutional blocks, batch normalization, and global average pooling, trained from scratch.

**Transfer learning**
Three ImageNet backbones: EfficientNetB0, ResNet50V2, and DenseNet121. Each is trained in two stages: a frozen backbone stage where only the new classification head is trained, followed by a fine tuning stage where the top layers of the backbone are unfrozen and trained at a lower learning rate.

**Hyperparameter tuning**
A Keras Tuner random search over dense layer size, dropout rate, and learning rate, run on the strongest backbone from the comparison above.

**Evaluation**
Every model, including the baseline, the three backbones, and the tuned final model, is scored on the same held out test split using accuracy, precision, recall, F1 score, parameter count, and inference time.

**Deployment**
The best performing model is served through a Gradio interface that accepts an uploaded image and returns predicted class probabilities.

## Engineering notes

- Class weighting is applied during training to account for the uneven number of images per disease category.
- Every training stage checkpoints to disk after each epoch, so an interrupted training run resumes from its last completed epoch rather than starting over.
- Datasets are cached to disk rather than held in memory, since caching a dataset of this size in RAM is a meaningful contributor to out of memory crashes on a shared Colab runtime.

## Results

**Best model: EfficientNetB0** (fine tuned)

| Metric | Score |
|---|---|
| Accuracy | 88.28% |
| Precision | 89.34% |
| Recall | 88.28% |
| F1 Score | 88.62% |

Full comparison across all five models (Custom CNN, EfficientNetB0, EfficientNetB0 tuned, ResNet50V2, DenseNet121):

| Model | Accuracy | Precision | Recall | F1 | Params | Inference (ms) |
|---|---|---|---|---|---|---|
| EfficientNetB0 | 88.28% | 89.34% | 88.28% | 88.62% | 4,379,049 | 12.50 |
| EfficientNetB0 (tuned) | 87.30% | 88.27% | 87.30% | 87.60% | 4,379,049 | 9.77 |
| Custom CNN | 86.66% | 88.29% | 86.66% | 86.99% | 457,670 | 4.82 |
| ResNet50V2 | 86.49% | 87.28% | 86.49% | 86.74% | 24,090,886 | 15.00 |
| DenseNet121 | 82.17% | 83.44% | 82.17% | 82.41% | 7,301,446 | 17.80 |

The untuned EfficientNetB0 backbone edged out the tuned version slightly on this test split, and was selected as the final deployed model on that basis. See the comparative analysis section in the notebook for training curves and per class confusion matrices, along with a written discussion of the results.

## Tech stack

TensorFlow / Keras, Keras Tuner, scikit-learn, pandas, Gradio

## Usage

1. Open the notebook in Google Colab.
2. Run the cells in order. The dataset downloads automatically via `kagglehub` on first run.
3. Training produces a saved `.keras` model for the custom CNN, each backbone, and the final tuned model.
4. The last cell launches a Gradio interface with a public link for testing the deployed model on new images.

