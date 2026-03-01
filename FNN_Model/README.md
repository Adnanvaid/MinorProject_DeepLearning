🍇 Date Fruit Classification using ANN
📌 Project Overview

This project implements an Feedforward Neural Network (FNN) to classify different varieties of date fruits based on their physical and morphological attributes.

The objective is to build a predictive model capable of automatically identifying the correct class of a date fruit using structured numerical features.


🧠 Problem Statement

Manual classification of agricultural products can be time-consuming and inconsistent.

This project addresses the problem of:

Automatically classifying date fruit varieties based on measurable attributes using a supervised learning approach.


🔎 Architecture Flow

Input Features
      ↓
Hidden Layer (ReLU)
      ↓
Hidden Layer (ReLU)
      ↓
Output Layer (Softmax / Logits)
      ↓
Predicted Class


⚙️ Implementation Details

| Component              | Description                          |
| ---------------------- | ------------------------------------ |
| 🧹 Data Preprocessing  | Feature scaling and train-test split |
| 🧠 Model Type          | Artificial Neural Network (ANN)      |
| 🔁 Activation Function | ReLU (hidden layers)                 |
| 🎯 Loss Function       | CrossEntropyLoss                     |
| ⚡ Optimizer            | Gradient-based optimization          |
| 📊 Evaluation          | Accuracy & loss tracking             |
