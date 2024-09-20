# Fine-tuning ResNet18 with Triplet Loss and Cross-Entropy Loss for Binary Classification on CIFAR-10: Airplanes vs Cars

## Introduction
This project involves fine-tuning a pre-trained ResNet18 model for binary classification tasks using a combination of **Triplet Loss** and **Cross-Entropy Loss**. This approach enhances the model's ability to distinguish between two classes effectively.

## Table of Contents
1. [Section 1: Training with Cross-Entropy Loss and Fine-Tuning the Classifier](#section-1-training-with-cross-entropy-loss-and-fine-tuning-the-classifier)
2. [Section 2: Training with Triplet Loss and Fine-Tuning the Classifier](#section-2-training-with-triplet-loss-and-fine-tuning-the-classifier)
3. [Section 3: Fine-Tuning ResNet18 with Combination of Triplet Loss and Cross-Entropy Loss](#section-3-fine-tuning-resnet18-with-combination-of-triplet-loss-and-cross-entropy-loss)
4. [Results and Visualization](#results-and-visualization)
5. [Model Saving and Reloading](#model-saving-and-reloading)
6. [Test Phase](#test-phase)

## Section 1: Training with Cross-Entropy Loss and Fine-Tuning the Classifier
In this section, we focus on training the ResNet18 model using **Cross-Entropy Loss** for binary classification. The model is initialized with pre-trained weights, and the final fully connected layer is modified to suit our task.

### Cross-Entropy Loss Training
- The ResNet18 architecture is adjusted for binary classification.
- The model is trained for a specified number of epochs using cross-entropy loss, monitoring validation accuracy and loss after each epoch.

## Section 2: Training with Triplet Loss and Fine-Tuning the Classifier
In this section, we leverage **Triplet Loss** to enhance feature extraction and fine-tune the ResNet18 model for binary classification.

### Triplet Loss Feature Extraction
- A custom `TripletLoss` class is defined, and the ResNet18 architecture is modified for feature extraction.
- The model is trained for 50 epochs using triplet loss. Each batch is split into anchor, positive, and negative examples, and the loss is calculated accordingly.

### Classifier Fine-Tuning
- The final fully connected layer is unfrozen, and the model is trained for 5 additional epochs using cross-entropy loss.
- Validation accuracy and loss are monitored after each epoch.

## Section 3: Fine-Tuning ResNet18 with Combination of Triplet Loss and Cross-Entropy Loss
In this section, we fine-tune a pre-trained ResNet18 model for binary classification using both triplet loss for feature extraction and cross-entropy loss for classification.

### Model Architecture and Modifications
- The ResNet18 model is initialized with a modified fully connected layer for binary classification.
- All parameters are set to be trainable.

### Custom Triplet Loss Function
A custom loss function combines triplet loss and cross-entropy loss to improve feature separation and classification accuracy.

### Training Process
- The model is trained for 10 epochs, combining both losses to update the model weights effectively.
- Validation accuracy is recorded after each epoch.

### Results and Visualization
- Accuracy and loss plots are generated, providing insights into the model's learning dynamics.

## Model Saving and Reloading
The trained model is saved to Google Drive for future use, enabling easy reloading for inference or further fine-tuning.

## Test Phase
After training, the model is evaluated on the test dataset. The final test accuracy is computed, demonstrating improved performance due to the effective integration of triplet loss and cross-entropy loss.

```python
# Example code for training and evaluation
net3 = models.resnet18(pretrained=True)
net3.fc = nn.Linear(net3.fc.in_features, 2)

# Training and evaluation code here...

print(f"Test Accuracy: {test_accuracy:.4f}")
```

## Results
The model achieves significant accuracy improvements, indicating the effectiveness of the combined loss functions in enhancing classification performance.

---
```

Feel free to modify any sections further if needed!
