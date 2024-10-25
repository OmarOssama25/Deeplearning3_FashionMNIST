# Fashion MNIST Classification using Keras

## Project Objective
This project implements a deep learning model to classify fashion items from the Fashion MNIST dataset using Keras with TensorFlow backend. The goal is to achieve over 90% accuracy in classifying 10 different categories of clothing items using a neural network architecture.

## Dataset Description
The Fashion MNIST dataset consists of:
- 70,000 grayscale images (28x28 pixels)
- 60,000 training images
- 10,000 testing images
- 10 fashion categories:
  * T-shirt/top
  * Trouser
  * Pullover
  * Dress
  * Coat
  * Sandal
  * Shirt
  * Sneaker
  * Bag
  * Ankle boot

## Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow >= 2.0.0
- numpy
- matplotlib
- seaborn
- scikit-learn
- pandas

## Installation Instructions
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fashion-mnist-classification.git
cd fashion-mnist-classification
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter Notebook:
```bash
jupyter notebook Fashion_MNIST_Keras.ipynb
```

2. Run all cells in sequence to:
   - Load and preprocess the data
   - Build and train the model
   - Evaluate performance
   - Visualize results

## Model Architecture
- Input Layer: 784 neurons (28x28 flattened)
- Hidden Layers:
  * Dense layer (256 neurons, ReLU activation)
  * Dense layer (128 neurons, ReLU activation)
  * Dense layer (64 neurons, ReLU activation)
- Output Layer: 10 neurons (Softmax activation)
- Regularization: Dropout

## Features
- Data normalization and preprocessing
- Learning rate scheduling with warmup
- Early stopping and model checkpointing
- Comprehensive visualization tools:
  * Training/validation curves
  * Confusion matrix
  * Classification metrics
  * ROC curves

## Model Performance
- Training Accuracy: ~90%
- Validation Accuracy: ~88.6%
- Test Accuracy: ~88.33%
- Per-class metrics available in classification report

## Results Visualization
The notebook includes:
- Loss and accuracy curves
- Confusion matrix
- Classification metrics by class

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Fashion MNIST dataset by Zalando Research
- TensorFlow and Keras documentation
- Deep learning community

## Contact
Your Name - [your.email@example.com]
Project Link: [https://github.com/yourusername/fashion-mnist-classification]

---
**Note**: This project is for educational purposes and demonstrates deep learning concepts using the Fashion MNIST dataset.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/29393552/9298eb38-8feb-4143-be91-99246e7d71a9/Fashion_MNIST_Keras.ipynb
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/29393552/24d8bb8b-5117-4389-b023-83de78a94959/Fashion_MNIST_Keras.ipynb
