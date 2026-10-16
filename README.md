# IMDB Movie Review Sentiment Analysis Using RNN and LSTM

## 📌 Project Overview

This project performs sentiment analysis on movie reviews using deep learning. It compares a **Simple Recurrent Neural Network (RNN)** with a **Bidirectional Long Short-Term Memory (LSTM)** model to classify movie reviews as positive or negative.

The models are trained and evaluated using the IMDB movie review dataset with TensorFlow and Keras.

## 🎯 Objectives

- Classify movie reviews as positive or negative.
- Preprocess text sequences for deep learning.
- Implement and compare Simple RNN and Bidirectional LSTM models.
- Evaluate models using accuracy, loss, precision, recall, and F1-score.
- Visualize model performance using training curves and confusion matrices.
- Predict sentiment for new movie reviews.

## 📂 Dataset

The project uses the built-in IMDB dataset from `tensorflow.keras.datasets.imdb`.

| Property | Description |
|---|---|
| Total reviews | 50,000 |
| Training reviews | 25,000 |
| Testing reviews | 25,000 |
| Number of classes | 2 |
| Negative label | 0 |
| Positive label | 1 |
| Vocabulary size | 10,000 most frequent words |
| Maximum sequence length | 200 tokens |

The dataset is loaded automatically through Keras, so a separate dataset download is not required.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Google Colab

## 🧠 Model Architecture

### 1. Simple RNN

The Simple RNN processes a sequence of tokens and maintains a hidden state to learn sequential patterns in the review.

**Architecture:**

Input Sequence → Embedding → SimpleRNN → Dropout → Dense Layer → Sigmoid Output

### 2. Bidirectional LSTM

The Bidirectional LSTM processes the review in both forward and backward directions, allowing it to learn contextual information from both directions.

**Architecture:**

Input Sequence → Embedding → Bidirectional LSTM → Dropout → Dense Layer → Sigmoid Output

### Common Configuration

- Vocabulary size: 10,000
- Maximum sequence length: 200
- Embedding dimension: 128
- Recurrent units: 64
- Dropout rate: 0.5
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Batch size: 64
- Maximum epochs: 5
- Early stopping based on validation loss

## ⚙️ Project Workflow

1. Load the IMDB movie review dataset.
2. Limit the vocabulary to the 10,000 most frequent words.
3. Pad or truncate reviews to a fixed length of 200 tokens.
4. Build the Simple RNN and Bidirectional LSTM models.
5. Compile and train the models.
6. Evaluate performance on the test dataset.
7. Generate classification reports and confusion matrices.
8. Plot training and validation accuracy and loss.
9. Predict sentiment for custom movie reviews.

## 📊 Results

The following results were obtained in the recorded experiment.

| Metric | Simple RNN | Bidirectional LSTM |
|---|---:|---:|
| Test accuracy | 54.33% | **84.92%** |
| Test loss | 0.6826 | **0.3720** |

### Simple RNN — Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.52 | 0.95 | 0.67 |
| Positive | 0.72 | 0.14 | 0.24 |
| **Overall accuracy** | | | **0.54** |

### Bidirectional LSTM — Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.88 | 0.81 | 0.84 |
| Positive | 0.83 | 0.88 | 0.85 |
| **Overall accuracy** | | | **0.85** |

### Key Findings

- The Bidirectional LSTM achieved **84.92% test accuracy**.
- Its test accuracy was **30.59 percentage points higher** than the Simple RNN in this experiment.
- The Simple RNN had low positive-class recall, while the LSTM performed more evenly across both sentiment classes.
- The LSTM achieved an F1-score of approximately 0.85 for both classes.

Results may vary slightly with different training runs or environments.

## 💬 Example Predictions

Sample output from the trained LSTM model:

| Movie Review | Predicted Sentiment | Positive Probability |
|---|---|---:|
| This movie was absolutely fantastic, the acting and story were brilliant! | Positive | 0.82 |
| Terrible film. Boring plot, awful acting, a complete waste of time. | Negative | 0.01 |
| It was okay, a few good moments but overall quite forgettable. | Negative | 0.34 |

The positive probability represents the model's estimated probability for the positive class. It is not a guarantee that the prediction is correct.

## 🚀 How to Run the Project

### Option 1: Google Colab (Recommended)

1. Download `imdb_sentiment_full.ipynb` from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select **File → Upload notebook**.
4. Upload the downloaded notebook.
5. Run all cells from top to bottom.

The notebook loads the IMDB dataset automatically. Training may take several minutes.

### Option 2: Run Locally

**Step 1: Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/imdb-movie-review-sentiment-analysis.git
```

**Step 2: Open the project folder**

```bash
cd imdb-movie-review-sentiment-analysis
```

**Step 3: Install the dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Open the notebook**

```bash
jupyter notebook
```

Open `imdb_sentiment_full.ipynb` and run the cells in order.

Make sure the notebook uses the Python environment in which the required packages are installed.

## 📁 Repository Structure

```text
imdb-movie-review-sentiment-analysis/
│
├── README.md
├── imdb_sentiment_full.ipynb
└── requirements.txt
```

## ⚠️ Limitations

- The project performs binary sentiment classification only.
- Sarcasm, irony, and mixed opinions may be difficult to classify.
- Model predictions depend on the quality and patterns of the training data.
- The recorded evaluation results may differ across training runs.

## 🔮 Future Improvements

- Experiment with GRU and other LSTM architectures.
- Tune model hyperparameters to improve performance.
- Explore pretrained word embeddings.
- Develop a web application for interactive sentiment prediction.
- Evaluate performance on additional review datasets.

## 📚 Learning Outcomes

This project demonstrates practical applications of natural language processing, recurrent neural networks, sequence preprocessing, binary text classification, and deep learning model evaluation.
