🤖 Build Your Own GPT from Scratch: Transformer Decoder Implementation
This repository hosts the code developed in the video "Visually understand and code GPT from scratch | Build it yourself using Google collab," which provides a step-by-step breakdown and implementation of a simplified Generative Pre-trained Transformer (GPT) model.
📖 Project Overview
The core objective of this project is twofold: first, to visually and mathematically unpack the Transformer Architecture, and second, to implement and train a simplified version of this model using PyTorch on Google Colab.
The model is designed to predict the next word/token iteratively, given a sequence of input tokens. The architecture utilizes the decoder stack of the Transformer model proposed in the paper "Attention Is All You Need," which is the core model architecture used by many modern large language models (LLMs) like Chat GPT.

🧠 Model Architecture & Key Components
The implementation focuses on coding a single Transformer Layer, which includes the following sequential components:
1. Input Embeddings and Positional Encoding (Token + Position Embedding): Each word/token in the sequence is first converted into a learnable embedding vector (e.g., length 4 initially). Positional Embeddings are then added element-wise to the token embeddings to give the model information about the position of the token within the input sequence.
2. Masked Multi-Head Attention: This is crucial for establishing relationships between each word and other words prior to it in the sequence.
    ◦ The attention mechanism calculates Query (Q), Key (K), and Value (V) matrices.
    ◦ It uses a scaled dot-product (QKᵀ / √dₖ) to normalize scores.
    ◦ Causal Masking (implemented using a large negative value) is applied before softmax to ensure that tokens do not interact with future tokens in the sequence.
3. Residual Connection + Layer Normalization (Add & Norm 1): The output of the multi-head attention layer is summed with the input from the previous step (residual connection), and then Layer Normalization is applied to help stabilize training.
4. Position-wise Feed Forward Network: This layer equips the model to learn nonlinearities and specific token characteristics (e.g., whether a token is a verb or a noun). It involves expanding the matrix and then condensing it back, using a ReLU activation function to avoid vanishing gradient problems.
5. Residual Connection + Layer Normalization (Add & Norm 2): A second layer of residual connection and layer normalization follows the feed-forward network.
6. Final Linear Layer and Softmax: The final normalized output is passed through a weight matrix (D x V, where D is embedding size and V is vocabulary size) to calculate logits. A final row-wise Softmax is applied to convert these logits into predicted probabilities for each token in the vocabulary.

⚙️ Setup and Prerequisites
The code is primarily structured for use within a Google Colab Notebook.
Requirements:
• PyTorch Library: Used for coding the architecture.
• Google Colab Runtime: Ensure you select a T4 GPU for the runtime type to execute the notebook efficiently, especially when scaling up the model.

📚 Data Set
The training is performed using a synthetic dataset to simplify the learning process:
• Data Content: All integers from 1 up to 999 written down in words (e.g., "one," "two," "one hundred twenty-one").
• Vocabulary Size: The unique vocabulary size is small, consisting of only 28 unique tokens.
The model is trained to take a sequence of tokens (inputs) and predict the subsequent token (target), which is effectively the input sequence shifted right by one.

🚀 Usage and Training
The repository includes code for training and generating text (inference).
Initial Simple Model Parameters
The initial simplified model coded aligns closely with the visual breakdown on Google Sheets:
• Sequence Length (Block Size): 10 tokens.
• Embedding Length (n_embeddings): 4.
• Number of Heads: 2.
• Number of Layers: 1.
• Total Parameters: Relatively small (e.g., 480 total parameters).
Scaled-Up Model Parameters
To achieve better prediction performance, the model was scaled up for deeper training. The scaled-up hyperparameters include:
• Sequence Length (Block Size): 20.
• Embedding Length (n_embeddings): 128.
• Number of Heads: 8.
• Number of Layers: 4 (four copies of attention + feed forward layers).
• Total Parameters: Increased significantly (e.g., approximately 7.3 lakhs).

Inference
Prediction is handled by the generate function. This function iteratively predicts the next token by:
1. Truncating the input sequence to the maximum context window (e.g., 10 or 20).
2. Calling the GPT forward function to get the final logits.
3. Applying Softmax to the last row of logits to obtain probabilities.
4. Selecting the token with the highest probability and concatenating it to the input sequence for the next iteration.

📝 Code Structure
Two different Google Colab notebooks are available, providing implementations of the architecture discussed:
1. Simple Script: A direct, procedural implementation matching the step-by-step coding section of the video.
2. Modular Notebook: A parallel notebook structured with classes and objects to make the code more modular, allowing models to be saved and reused.
