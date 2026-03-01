🧠 CIFAR-10 Image Classification API (CNN + FastAPI)
📌 Project Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch for image classification on the CIFAR-10 dataset. The trained model is deployed using FastAPI, allowing users to upload an image via a REST API and receive real-time predictions.

The project demonstrates an end-to-end deep learning workflow, including model training, serialization, API integration, and inference deployment.


🚀 Features

CNN model trained on CIFAR-10
FastAPI-based REST API
Image upload support using multipart form data
Real-time prediction endpoint
Production-ready project structure


🛠 Tech Stack

Python
PyTorch
FastAPI
Uvicorn


⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/Adnanvaid/MinorProject_DeepLearning.git
cd  MinorProject_DeepLearning/CNN_Model

2️⃣ Create Virtual Environment
python -m venv env

Activate the environment:

Windows: env\Scripts\activate
Mac/Linux: source env/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
uvicorn api:app --reload

📌 Example Categories (CIFAR-10)

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck