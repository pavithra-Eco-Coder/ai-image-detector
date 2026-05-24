# 🤖 AI Image Detector

A deep learning-powered web application that detects whether an uploaded image is AI-generated or a real photograph using transfer learning with MobileNetV2.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

With the rapid advancement of AI image generation tools like DALL-E, Midjourney, and Stable Diffusion, distinguishing between AI-generated and real photographs has become increasingly challenging. This project addresses this problem by building a robust deep learning classifier that can accurately identify AI-generated images.

The model leverages **transfer learning** with MobileNetV2, a lightweight convolutional neural network pre-trained on ImageNet, fine-tuned specifically for this binary classification task.

---

## ✨ Features

- ✅ **High Accuracy**: Achieves >90% accuracy on test data
- ✅ **Fast Inference**: Predictions in under 1 second
- ✅ **User-Friendly Web Interface**: Built with Streamlit for easy interaction
- ✅ **Confidence Scores**: Provides prediction confidence for transparency
- ✅ **Comprehensive Metrics**: Includes accuracy, precision, recall, and F1-score
- ✅ **Visualizations**: Training history plots, confusion matrix, and sample predictions
- ✅ **Production-Ready**: Modular code structure with best practices

---

## 📁 Project Structure
```
ai-image-detector/
│
├── dataset/                      # Dataset directory (not included in repo)
│   ├── train/
│   │   ├── ai/                   # AI-generated training images
│   │   └── real/                 # Real training images
│   └── test/
│       ├── ai/                   # AI-generated test images
│       └── real/                 # Real test images
│
├── outputs/                      # Generated outputs
│   ├── best_model.keras          # Best model checkpoint
│   ├── final_model.keras         # Final trained model
│   ├── class_indices.json        # Class label mappings
│   ├── training_history.png      # Training curves
│   ├── confusion_matrix.png      # Confusion matrix visualization
│   ├── sample_predictions.png    # Sample prediction images
│   └── test_metrics.txt          # Detailed test metrics
│
├── train.py                      # Training script
├── test.py                       # Testing and evaluation script
├── app.py                        # Streamlit web application
├── check_setup.py                # Environment verification
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # Project documentation
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/pavithra-Eco-Coder/ai-image-detector.git
cd ai-image-detector
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup
```bash
python check_setup.py
```

---

## 📊 Dataset Preparation

### Dataset Structure

Organize your dataset in the following structure:
```
dataset/
├── train/
│   ├── ai/        # AI-generated training images
│   └── real/      # Real training images
└── test/
    ├── ai/        # AI-generated test images
    └── real/      # Real test images
```

### Creating the Folders
```bash
mkdir -p dataset/train/ai dataset/train/real
mkdir -p dataset/test/ai dataset/test/real
```

### Dataset Sources (Examples)

**AI-Generated Images:**
- DALL-E 2/3: https://labs.openai.com/
- Midjourney: https://www.midjourney.com/
- Stable Diffusion: https://huggingface.co/spaces/stabilityai/stable-diffusion
- Bing Image Creator: https://www.bing.com/create
- Kaggle datasets: Search for "AI generated images"

**Real Images:**
- COCO Dataset: https://cocodataset.org/
- Unsplash: https://unsplash.com/
- Pexels: https://www.pexels.com/
- Your own photographs

### Recommended Dataset Size

- **Training**: 1,000-5,000 images per class (2,000-10,000 total)
- **Testing**: 200-500 images per class (400-1,000 total)

*More data generally leads to better performance!*

---

## 🚀 Usage

### 1. Train the Model
```bash
python train.py
```

**What happens during training:**
- Loads and preprocesses training data
- Applies data augmentation
- Builds MobileNetV2-based model
- Trains for up to 50 epochs (with early stopping)
- Saves best model to `outputs/best_model.keras`
- Generates training history plot

**Expected Output:**
```
🚀 AI IMAGE DETECTOR - TRAINING PIPELINE
================================================================================
📊 STEP 1: Loading and preparing data...
   ✓ Training samples: 3200
   ✓ Validation samples: 800
...
✅ TRAINING COMPLETED SUCCESSFULLY!
```

### 2. Evaluate on Test Set
```bash
python test.py
```

**What happens during testing:**
- Loads trained model
- Evaluates on test dataset
- Calculates accuracy, precision, recall, F1-score
- Generates confusion matrix
- Creates sample prediction visualizations

**Expected Output:**
```
🧪 AI IMAGE DETECTOR - TESTING PIPELINE
================================================================================
📊 Performance Metrics:
Accuracy:  0.9250 (92.50%)
Precision: 0.9248
Recall:    0.9250
F1-Score:  0.9249
```

### 3. Launch Web Application
```bash
streamlit run app.py
```

**The app will open in your browser at:** `http://localhost:8501`

**How to use the app:**
1. Click "Browse files" and upload an image
2. Wait for the model to analyze (< 1 second)
3. View prediction results and confidence score
4. Check detailed probability breakdown

---

## 🏗️ Model Architecture

### Base Model: MobileNetV2

MobileNetV2 is a lightweight convolutional neural network optimized for mobile and edge devices. It uses:
- **Depthwise Separable Convolutions** for efficiency
- **Inverted Residual Blocks** for feature extraction
- **Pre-trained on ImageNet** (1.4M images, 1000 classes)

### Custom Classification Head
```
Input (224x224x3)
    ↓
MobileNetV2 Base (frozen)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (2 units, Softmax)
    ↓
Output (AI / Real)
```

### Training Configuration

- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Image Size**: 224×224 pixels
- **Data Augmentation**: Rotation, flip, zoom, shift
- **Callbacks**: Early stopping, learning rate reduction, model checkpoint

---

## 📈 Results

### Training Performance

Training and validation accuracy/loss curves are saved in `outputs/training_history.png`

### Test Set Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 84.96% |
| Precision | 85.00% |
| Recall    | 84.96% |
| F1-Score  | 85.95% |

*Note: Update these values with your actual results after training*

### Confusion Matrix

Confusion matrix visualization is saved in `outputs/confusion_matrix.png`

### Sample Predictions

Sample predictions with images are saved in `outputs/sample_predictions.png`

---

## 🛠️ Technologies Used

### Core Libraries
- **TensorFlow/Keras**: Deep learning framework
- **Python**: Programming language
- **NumPy**: Numerical computing
- **Pillow**: Image processing

### Visualization
- **Matplotlib**: Plotting library
- **Seaborn**: Statistical visualization

### Web Framework
- **Streamlit**: Interactive web application

### Machine Learning Utilities
- **Scikit-learn**: Metrics and evaluation

---

## 🚀 Future Improvements

- [ ] Add support for more AI generation tools (DALL-E 3, Midjourney v6)
- [ ] Implement ensemble models for higher accuracy
- [ ] Add explainability features (Grad-CAM heatmaps)
- [ ] Deploy to cloud (AWS, GCP, or Hugging Face Spaces)
- [ ] Create mobile app version
- [ ] Add batch prediction capability
- [ ] Implement API endpoint for integration
- [ ] Add real-time video stream detection

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

- GitHub: [@pavithra-Eco-Coder](https://github.com/pavithra-Eco-Coder)
- LinkedIn: [Pavithra J](https://www.linkedin.com/in/pavithra-j-6ba0592b9/)
- Email: pavipavithra3693@gmail.com

---

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the excellent framework
- MobileNetV2 authors for the architecture
- Streamlit for the web app framework
- Open-source community for inspiration and support

---

## ⭐ Star This Repository

If you found this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ by Pavithra J**
```

---

