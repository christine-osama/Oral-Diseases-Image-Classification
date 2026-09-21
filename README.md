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

See the comparative analysis section in the notebook for the full metrics table, training curves, and confusion matrices for each model, along with a written discussion of the results.

## Tech stack

TensorFlow / Keras, Keras Tuner, scikit-learn, pandas, Gradio

## Usage

1. Open the notebook in Google Colab.
2. Run the cells in order. The dataset downloads automatically via `kagglehub` on first run.
3. Training produces a saved `.keras` model for the custom CNN, each backbone, and the final tuned model.
4. The last cell launches a Gradio interface with a public link for testing the deployed model on new images.

## License

This project is provided for educational purposes.
