# 🧠 CNN Handwritten Digit Classifier

A comprehensive Convolutional Neural Network (CNN) project that recognizes handwritten digits using the MNIST dataset. Perfect for beginners learning deep learning and computer vision!

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project implements a deep learning CNN that achieves **~99% accuracy** on digit recognition. It includes detailed comments and visualizations to help you understand exactly how CNNs work under the hood.

### What You'll Learn:
- How Convolutional Neural Networks process images
- Feature extraction using convolutional layers
- The role of pooling, dropout, and batch normalization
- Training and evaluating deep learning models
- Visualizing learned features and predictions

## 📊 Model Architecture

```
Input (28x28x1 grayscale image)
    ↓
Conv2D (32 filters, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU + BatchNorm
    ↓
Flatten
    ↓
Dense (128 units) + ReLU + Dropout(0.5)
    ↓
Dense (64 units) + ReLU + Dropout(0.3)
    ↓
Dense (10 units) + Softmax
    ↓
Output (Digit 0-9 with confidence)
```

**Total Parameters:** ~1.2M  
**Training Time:** ~5-10 minutes on Colab (with GPU)  
**Test Accuracy:** 98-99%

## 🚀 Quick Start

### Option 1: Google Colab (Recommended - No Installation Required!)

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy and paste the entire code
4. Click **Runtime → Run all**
5. Watch the magic happen! ✨

### Option 2: Local Installation

**Prerequisites:**
- Python 3.7 or higher
- pip package manager

**Installation:**

```bash
# Clone the repository (or download the code)
git clone https://github.com/yourusername/cnn-digit-classifier.git
cd cnn-digit-classifier

# Install required libraries
pip install tensorflow numpy matplotlib

# Run the script
python cnn_mnist_classifier.py
```

## 📦 Required Libraries

```python
tensorflow>=2.0.0    # Deep learning framework
numpy>=1.19.0        # Numerical computing
matplotlib>=3.3.0    # Visualization
```

**Note:** Google Colab has all these pre-installed!

## 🎨 Features

### 1. **Comprehensive Dataset Visualization**
- View sample training images
- Understand data distribution
- See preprocessing steps

### 2. **Detailed Model Architecture**
- Layer-by-layer explanation
- Parameter counts
- Visual model summary

### 3. **Training Visualization**
- Real-time accuracy tracking
- Loss curves over epochs
- Validation performance monitoring

### 4. **Prediction Analysis**
- Confidence scores for each digit
- Color-coded correct/incorrect predictions
- Probability distribution visualization

### 5. **Feature Visualization**
- View learned convolutional filters
- Understand what the CNN "sees"
- Feature maps at different layers

### 6. **Model Saving**
- Save trained model for later use
- Easy model deployment

## 📈 Results

| Metric | Score |
|--------|-------|
| Training Accuracy | ~99.5% |
| Validation Accuracy | ~99.2% |
| Test Accuracy | ~99.0% |
| Training Time | 5-10 min (GPU) |
| Model Size | ~4.5 MB |

## 🖼️ Sample Output

The script generates several visualizations:

1. **Training Images**: See what the model learns from
2. **Training History**: Accuracy and loss curves
3. **Predictions**: Visual comparison of predictions vs. true labels
4. **Learned Filters**: What patterns the CNN detects
5. **Confidence Bars**: Probability distribution for each digit

## 🧩 How It Works

### Step-by-Step Process:

1. **Load MNIST Dataset**
   - 60,000 training images
   - 10,000 test images
   - Each image is 28x28 pixels

2. **Preprocess Data**
   - Reshape to (28, 28, 1) for CNN input
   - Normalize pixel values to [0, 1]
   - Convert labels to one-hot encoding

3. **Build CNN Model**
   - Convolutional layers extract features
   - Pooling layers reduce dimensions
   - Dense layers perform classification

4. **Train the Model**
   - Batch size: 128
   - Epochs: 10
   - Optimizer: Adam
   - Loss: Categorical Crossentropy

5. **Evaluate Performance**
   - Test on unseen data
   - Calculate accuracy and loss
   - Visualize predictions

6. **Analyze Results**
   - View learned filters
   - Check confidence scores
   - Identify misclassifications

## 🎓 Educational Value

This project is perfect for:
- **Beginners**: Extensive comments explain every line
- **Students**: Understand CNN concepts with visuals
- **Educators**: Use as teaching material
- **Researchers**: Baseline for digit recognition tasks

### Key Concepts Covered:

- ✅ Convolutional layers and filters
- ✅ Pooling (MaxPooling)
- ✅ Batch Normalization
- ✅ Dropout regularization
- ✅ Activation functions (ReLU, Softmax)
- ✅ One-hot encoding
- ✅ Training/validation split
- ✅ Model evaluation metrics
- ✅ Feature visualization

## 🛠️ Customization

### Modify the Architecture:

```python
# Add more convolutional layers
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

# Change dropout rate
model.add(layers.Dropout(0.4))  # Default is 0.5

# Adjust number of neurons
model.add(layers.Dense(256, activation='relu'))  # Default is 128
```

### Hyperparameter Tuning:

```python
# Increase training epochs
history = model.fit(X_train, y_train, epochs=20)  # Default is 10

# Change batch size
history = model.fit(X_train, y_train, batch_size=64)  # Default is 128

# Try different optimizers
model.compile(optimizer='sgd', ...)  # Default is 'adam'
```

## 📝 Code Structure

```
cnn_mnist_classifier.py
├── Import Libraries
├── Load MNIST Dataset
├── Data Preprocessing
├── Build CNN Model
│   ├── Convolutional Blocks
│   ├── Flatten Layer
│   └── Dense Layers
├── Compile Model
├── Train Model
├── Visualize Training History
├── Evaluate on Test Data
├── Make Predictions
├── Visualize Learned Filters
├── Detailed Prediction Analysis
└── Save Model
```

## 🐛 Troubleshooting

### Common Issues:

**1. ImportError: No module named 'tensorflow'**
```bash
pip install tensorflow
```

**2. ResourceExhaustedError (Out of memory)**
```python
# Reduce batch size
history = model.fit(X_train, y_train, batch_size=64)
```

**3. Slow training on CPU**
- Use Google Colab with GPU: Runtime → Change runtime type → GPU

**4. Model not improving**
- Increase epochs
- Try different learning rates
- Check data preprocessing

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Resources

### Learn More About CNNs:
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [CS231n: CNN for Visual Recognition](http://cs231n.stanford.edu/)
- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
- [MNIST Dataset Documentation](http://yann.lecun.com/exdb/mnist/)

### Related Projects:
- Fashion MNIST Classification
- CIFAR-10 Image Recognition
- Custom Digit Recognition with Camera

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🌟 Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow and Keras teams
- Google Colab for free GPU resources
- The amazing deep learning community

## 📊 Project Stats

⭐ **Star this repo** if you found it helpful!  
🐛 **Report issues** to help improve the project  
🔄 **Fork and customize** for your own experiments

---

<div align="center">

**Made with ❤️ and Python**

If this project helped you learn CNNs, please consider giving it a ⭐!

</div>

---

## 🎯 Next Steps

After mastering this project, try:

1. **Fashion MNIST**: Classify clothing items
2. **CIFAR-10**: Recognize color images
3. **Custom Dataset**: Train on your own images
4. **Transfer Learning**: Use pre-trained models
5. **Data Augmentation**: Improve accuracy with augmented data

Happy Learning! 🚀
