# MLPR: Texture-Based Image Classification Pipeline

This project implements a machine learning pipeline to classify images into three texture classes—grass, sand, and stairs—based on their color and textural properties. The pipeline extracts first-order color features and second-order Gray-Level Co-occurrence Matrix (GLCM) features, and uses nested cross-validation to tune and evaluate models.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Visualization](#visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Nested Cross-Validation](#nested-cross-validation)
- [Results and Discussion](#results-and-discussion)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [References](#references)

## Introduction

The goal of this project is to develop a machine learning model capable of classifying images into one of three texture categories. Image URLs are provided in text files (for grass, sand, and stairs), and images undergo processing before feature extraction and model evaluation.

## Project Structure

- **MLPR25_exercise_Iftikhar_Momina.ipynb**: Jupyter Notebook containing the complete pipeline.
- **Data Files**: Text files (`grass.txt`, `sand.txt`, `stairs.txt`) containing image URLs.
- **Visualizations & Outputs**: Sample images, pairplots, histograms, PCA plots, confusion matrices, and summary tables from GridSearchCV and nested CV.

## Data Preparation

The dataset comprises image URLs divided into three categories:
- **Grass**
- **Sand**
- **Stairs**

### Image Processing Steps
1. **Resize:** Images are resized to a consistent dimension (e.g., 128×128 or 256×256 pixels).
2. **Grayscale Conversion:** Converted to grayscale for texture analysis.
3. **Quantization:** Grayscale images are quantized to 8 levels (0–7) to enhance GLCM computation.

## Methodology

### Data Preprocessing
- **Drive Mounting:** Uses Google Colab to mount Google Drive and access text files.
- **Image Processing:** Functions are provided to download, resize, convert to grayscale, and quantize each image.

### Feature Extraction

1. **First Order Color Features:**  
   - Computes mean and variance for each RGB channel.
   
2. **Second Order Texture Features:**  
   - Extracts GLCM features by calculating the correlation for pixel distances of 1 and 2 and angles of 0° and 90°.  
   - The chosen distances capture both local fine textures and broader texture patterns.

### Visualization

- **Pairplots:** To visualize relationships between standardized GLCM features.
- **Histograms:** Overlaid histograms for each Z-scored feature across classes.
- **PCA:** Principal Component Analysis is used to visualize clusters and assess class separability.

## Model Training and Evaluation

### Hyperparameter Tuning

Three classifiers are tuned using GridSearchCV with 5-fold stratified cross-validation:

1. **Ridge Classifier:**  
   - **Parameter:** L2 regularization strength (`alpha` values: 0.001, 0.01, 0.1, 1.0)  
   - **Best Result:** `alpha=0.001` with an accuracy of ~62.7%.

2. **Random Forest Classifier:**  
   - **Parameters:**
     - `n_estimators`: 100 to 300 (step 50)
     - `max_features`: `'sqrt'`, `'log2'`, or `None`
     - `bootstrap`: True/False  
   - **Best Result:** e.g., `{bootstrap=True, max_features=None, n_estimators=100}` with ~55.1% accuracy.
   - **Feature Importances:** Corr_Dist1_Angle0 is typically the most significant.

3. **Multi-Layer Perceptron (MLP) Classifier:**  
   - **Parameters:**
     - Hidden layer sizes from 15 to 40 neurons (step 5)
     - Activation functions: `tanh` and `relu`
     - Solvers: `sgd` and `adam`
     - Regularization (`alpha`): [0.01, 0.1, 1]
     - Validation fraction: 0.1 and 0.3  
   - **Best Result:** e.g., `{hidden_layer_sizes: (30,), activation: tanh, solver: adam, validation_fraction: 0.1, alpha: 0.1}` with ~63.2% accuracy.

### Nested Cross-Validation

Nested cross-validation is used for unbiased performance estimation:
- **Inner Loop:** 4-fold stratified cross-validation for hyperparameter tuning.
- **Outer Loop:** 5-fold stratified cross-validation for performance estimation.

The evaluation reports:
- Best hyperparameters per outer fold.
- Accuracy on each fold and an overall mean accuracy.
- Confusion matrices.
- For Random Forest, additional feature importances.

## Results and Discussion

- **Ridge Classifier:**  
  - Mean outer accuracy: ~62.7%  
  - Sample Confusion Matrix:
    ```
    [[39, 18, 5],
     [ 4, 44, 13],
     [16, 13, 33]]
    ```
  
- **Random Forest:**  
  - Mean outer accuracy: ~53.5%  
  - Key Feature: Corr_Dist1_Angle0.

- **MLP Classifier:**  
  - Mean outer accuracy: ~58.4%

### Discussion

- **Best Model:** The Ridge Classifier performs best with the current feature set, indicating that a simple linear model with L2 regularization can effectively discriminate between textures.
- **Limitations:**  
  - Moderate accuracy suggests that GLCM features alone might not capture complex textures fully.
  - Variability in image resolution and dataset size may affect performance.
- **Improvements:**  
  - Incorporate additional features (e.g., deeper texture features or color-texture fusion).
  - Use data augmentation to expand and diversify the dataset.
  - Experiment with advanced deep learning approaches.

## Installation and Setup

1. **Mount Google Drive:**  
   In Google Colab, mount your drive to access data files.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
## Install Required Libraries

Use the command below to install the necessary packages:

    pip install numpy pandas matplotlib scikit-image scikit-learn seaborn

## Ensure Data Files Availability

Place `grass.txt`, `sand.txt`, and `stairs.txt` in the specified directory (e.g., `/content/drive/MyDrive/MLPR/`).

## Usage

1. **Open the Notebook:**  
   Open `Machine learning and pattern recognition.ipynb` in Google Colab.
2. **Run the Notebook Cells Sequentially:**
   - Data loading and preprocessing.
   - Image processing (resize, grayscale, quantization).
   - Feature extraction and visualization.
   - Hyperparameter tuning and nested cross-validation.
   - Analysis of output plots and evaluation metrics.

## Dependencies

- **Python:** 3.x
- **Libraries:**
  - [numpy](https://numpy.org/)
  - [pandas](https://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [scikit-image](https://scikit-image.org/)
  - [scikit-learn](https://scikit-learn.org/)
  - [seaborn](https://seaborn.pydata.org/)

## References

- [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [StratifiedKFold Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [graycomatrix Documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix)
- [graycoprops Documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops)
- [RidgeClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
- [MLPClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [RandomForestClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [StandardScaler Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
