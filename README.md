# 🎭 Facial Emotion Recognition using CNN

A deep learning project that classifies **human facial expressions into seven emotion categories** using a **CNN-based transfer learning approach (DenseNet169)**.  
The project demonstrates a **complete end-to-end computer vision pipeline**: data preprocessing, model training, fine-tuning, and evaluation.

---

## 📌 Project Overview

Facial emotion recognition is a key problem in computer vision with applications in:
- Human–Computer Interaction
- Mental health analysis
- Smart surveillance systems
- Affective computing

In this project, a **Convolutional Neural Network (CNN)** is trained to recognize facial emotions from images using **transfer learning**, ensuring better generalization and faster convergence compared to training from scratch.

---

## 😊 Emotion Classes

The model classifies facial expressions into **7 categories**:

- 😠 Anger  
- 🤢 Disgust  
- 😨 Fear  
- 😊 Happy  
- 😐 Neutral  
- 😔 Sadness  
- 😲 Surprise  

---

## 🧠 Model Architecture

- **Backbone:** DenseNet169 (pretrained on ImageNet)
- **Type:** CNN-based Transfer Learning
- **Input Size:** 48 × 48 RGB images
- **Classifier Head:**
  - Global Average Pooling
  - Fully Connected Layers
  - Dropout for regularization
  - Softmax output (7 classes)

### 🔁 Training Strategy
The model is trained in **two phases**:
1. **Feature Extraction**  
   - DenseNet backbone frozen  
   - Only classifier head trained  

2. **Fine-Tuning**  
   - Backbone unfrozen  
   - Very low learning rate  
   - Improves emotion-specific feature learning  

---

## 🗂️ Dataset

- **Dataset Name:** FER-2013 (Facial Expression Recognition 2013)
- **Image Size:** 48 × 48
- **Type:** Facial expression images
- **Source:** Kaggle

> ⚠️ Due to size constraints, the dataset is **not included** in this repository.

### 📥 Dataset Download
You can download the dataset from:
https://www.kaggle.com/datasets/msambare/fer2013


## After extraction, the directory structure should be:
- project-root/
- ├── train/
- │ ├── angry/
- │ ├── disgust/
- │ ├── fear/
- │ ├── happy/
- │ ├── neutral/
- │ ├── sad/
- │ └── surprise/
- └── test/
- ├── angry/
- ├── disgust/
- ├── fear/
- ├── happy/
- ├── neutral/
- ├── sad/
- └── surprise/


---

## ⚙️ Data Preprocessing & Augmentation

To improve generalization, the following techniques are applied:
- Horizontal flipping
- Width and height shifting
- DenseNet-specific preprocessing
- Train–validation split

Class imbalance is handled using **class-weighted loss**.

---

## 📊 Evaluation Metrics

The model is evaluated using:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**
- **ROC–AUC (Multi-class)**

### 📈 Training Curves
- Training vs Validation Accuracy  
- Training vs Validation Loss  

> Typical validation accuracy for FER-2013 lies between **60–70%**, which is considered strong performance for this dataset.

---

## 📁 Project Structure
- Facial-Emotion-Recognition-CNN/
- │
- ├── src/
- │ ├── config.py
- │ ├── data_generator.py
- │ ├── model.py
- │ ├── train.py
- │ ├── evaluation.py
- │ └── main.py
- │
- ├── notebooks/
- │ └── CNNSentimentAnalysis.ipynb
- │
- ├── results/
- │ ├── accuracy_plot.png
- │ ├── loss_plot.png
- │ └── confusion_matrix.png
- │
- ├── models/
- │ └── README.md
- │
- ├── requirements.txt
- ├── .gitignore
- └── README.md


---

## ▶️ How to Run the Project

### 🚀 Google Colab (Recommended)

1. Open the notebook from `notebooks/` in Google Colab  
2. Upload and extract the dataset ZIP:
   ```bash
   unzip archive.zip
3. Ensure train/ and test/ folders are present
4. Run all cells sequentially
💡 Training is recommended on Colab for GPU acceleration.

### 💻 Run Locally (Optional)
- git clone https://github.com/Kkj1203/Facial-Emotion-Recognition-CNN.git
- cd Facial-Emotion-Recognition-CNN
- pip install -r requirements.txt
- python src/main.py

###⚠️ Training locally may be slow without GPU support.

## 📦 Model Weights
- Trained model files are not included due to file size limitations.
- You can generate them by running the training pipeline.

## 🧪 Key Learnings
- Practical use of CNNs for real-world vision problems
- Transfer learning and fine-tuning strategies
- Handling class imbalance in deep learning
- Interpreting evaluation metrics beyond accuracy

## 🛠️ Tech Stack
- Python 🐍
- TensorFlow / Keras
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

## 📌 Resume Description (Ready to Use)
- Facial Emotion Recognition using CNN
- Built a CNN-based deep learning model using DenseNet169 to classify facial expressions into seven emotion categories
- Applied image preprocessing, data augmentation, and class-weighted training to improve generalization
- Evaluated performance using accuracy, confusion matrix, and ROC–AUC metrics

## 🙌 Acknowledgements
- FER-2013 Dataset (Kaggle)
- TensorFlow & Keras documentation

## ⭐ If you found this project helpful, consider giving it a star!
