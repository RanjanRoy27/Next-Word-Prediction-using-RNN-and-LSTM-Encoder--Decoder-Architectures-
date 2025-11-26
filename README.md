Next-Word Prediction Using RNN and LSTM

A deep learning project comparing RNN and LSTM encoder–decoder architectures for next-word prediction using PyTorch.

Project Overview

This project builds and evaluates two sequence-modeling architectures—RNN and LSTM—for predicting the next word given a sequence of five input words.

The study includes:

dataset preprocessing

vocabulary creation

sequence generation

model training

evaluation (accuracy, perplexity, gradients, confusion matrix)

comparison of multi-layer RNN vs LSTM

This project was developed as part of an NLP coursework assignment.


Technologies Used

Python 3

PyTorch

TorchText

NumPy, Matplotlib

Scikit-learn (confusion matrix)

🧹 Dataset Preprocessing

Tokenization using basic_english tokenizer (TorchText)

Removed sentences with fewer than 5 tokens

Built vocabulary of top 5000 words

Created (input_sequence → target_word) pairs

Unknown words converted to <unk>

🏗️ Model Architectures
RNN Encoder–Decoder
Input → Embedding → 3-Layer RNN → Hidden State → Linear → Prediction

LSTM Encoder–Decoder
Input → Embedding → 3-Layer LSTM → Hidden State → Linear → Prediction

📊 Evaluation Metrics

Accuracy

Perplexity

Vanishing Gradient Visualization

Confusion Matrix (Top 30 Misclassifications)

Layer-wise Performance Comparison

OOV Word Evaluation

📝 Key Results
Model	Accuracy	Perplexity
RNN	~62%	~39.0
LSTM	~70%	~11.0

✔ LSTM converges faster
✔ Handles long-term dependencies better
✔ Lower perplexity
✔ Reduces vanishing gradient effect

🚀 How to Run the Project
1. Install Dependencies
pip install torch torchtext numpy matplotlib scikit-learn

2. Prepare Dataset

Place your text file here:

data/test.txt

3. Train Model
python src/train.py --model rnn
python src/train.py --model lstm

4. Evaluate
python src/evaluate.py

📌 Conclusion

This project demonstrates how RNN and LSTM architectures perform on real NLP sequence tasks.
LSTM clearly outperforms RNN due to improved handling of long-term dependencies.
