# LLM Output Bias Detection

A production-grade machine learning system for detecting stereotypical bias in LLM-generated text using the StereoSet benchmark dataset and fine-tuned MiniLM model.

## 🎯 Problem Statement

Large Language Models (LLMs) can inadvertently generate text that reinforces harmful stereotypes about various demographic groups. This project implements a bias detection system that classifies whether an LLM's response:

- **Reinforces stereotypes** (stereotype)
- **Challenges stereotypes** (anti-stereotype)
- **Is unrelated** to the stereotypical context (unrelated)

This capability is essential for building responsible AI systems and ensuring safe deployment of language models in production environments.

## 🔬 Why This Matters

1. **AI Safety**: Preventing the amplification of societal biases through AI systems
2. **Regulatory Compliance**: Meeting emerging AI ethics and fairness requirements
3. **User Trust**: Building AI products that users can trust to be fair and unbiased
4. **Responsible Development**: Enabling developers to audit LLM outputs before deployment

## 📊 Dataset

**StereoSet** (McGill-NLP) - A crowd-sourced benchmark for measuring stereotypical bias in language models.

| Attribute | Value |
|-----------|-------|
| Source | HuggingFace: `McGill-NLP/stereoset` |
| Subset | `intersentence` |
| Total Samples | 6,369 |
| Classes | 3 (balanced) |
| Bias Types | Race, Gender, Profession, Religion |

Each sample consists of a context sentence and multiple candidate continuations with gold labels indicating stereotype, anti-stereotype, or unrelated classifications.

## 🏗️ Modeling Approaches

### 1. Logistic Regression (Classical Baseline)
- **Features**: TF-IDF vectorization (10K features, uni/bi-grams)
- **Model**: Multinomial Logistic Regression with balanced class weights
- **Purpose**: Establish a classical ML baseline for comparison

### 2. Frozen MiniLM + Linear Head
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Approach**: Frozen encoder with trainable classification head
- **Pooling**: CLS token

### 3. LoRA Fine-tuned MiniLM (Main Model)
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Approach**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters
- **LoRA Config**: r=8, alpha=32, dropout=0.1, target=query,value
- **Pooling**: Mean pooling
- **Trainable Parameters**: 73,728 (0.32% of total)

⚠️Note: The frozen MiniLM and LoRA MiniLM are trained on google colab for GPU access and then downloaded for further usage in local system. The colab notebooks for both training both models are available in the notebooks folder and can be downloaded for personal use.

### Why LoRA/PEFT?

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by:
- Training only 0.32% of parameters while achieving superior performance
- Preserving the pre-trained knowledge of the base model
- Enabling quick adaptation to new domains
- Reducing storage requirements (only adapter weights need to be saved)

## 📈 Evaluation Results

| Model | Accuracy | F1 Score(Macro) |
|-------|----------|-----------|
| Logistic Regression | 0.3944 | 0.3939 |
| Frozen MiniLM | 54.29% | 54.21% |
| **LoRA MiniLM** | **75.00%** | **74.56%** |


The LoRA fine-tuned model achieves a **20+ percentage point improvement** over the frozen baseline, demonstrating the effectiveness of parameter-efficient fine-tuning for this task.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-output-bias-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your HuggingFace token:

```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your token
HF_TOKEN=your_huggingface_token_here
```

> **Note**: A HuggingFace token is required to download the StereoSet dataset. Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Train Logistic Regression Baseline

```bash
python -m src.models.logistic_regression
```

### Run the Streamlit Demo

```bash
streamlit run streamlit_app.py
```

The demo will be available at `http://localhost:8501`

## 📁 Project Structure

```
llm-output-bias-detection/
│
├── models/                     # Saved model weights
│   ├── logistic_regression/    # TF-IDF vectorizer + LogReg
│   ├── frozen_minilm/          # Frozen encoder + classifier
│   └── lora_minilm/            # LoRA adapters + classifier head
│       ├── adapter/            # PEFT adapter weights
│       └── classifier.pt       # Linear head weights
│
├── notebooks/                  # Training notebooks (reference only)
│   ├── frozen_minilm.ipynb     # Frozen baseline training
│   └── lora_minilm.ipynb       # LoRA fine-tuning
│
├── src/                        # Source code
│   ├── load_data.py            # StereoSet data loading
│   ├── preprocess.py           # Data preprocessing and splits
│   │
│   ├── models/                 # Model implementations
│   │   ├── logistic_regression.py
│   │   ├── frozen_minilm.py
│   │   └── lora_minilm.py
│   │
│   ├── evaluation/             # Evaluation utilities
│   │   └── metrics.py
│   │
│   ├── inference/              # Production inference
│   │   └── predictor.py
│   │
│   └── utils/                  # Utilities
│       ├── config.py
│       └── logger.py
│
├── streamlit_app.py            # Demo application
├── requirements.txt            # Dependencies
├── .env.example                # Environment template
└── README.md                   # This file
```

## 🔧 Usage

### Python API

```python
from src.inference.predictor import BiasPredictor

# Initialize predictor (loads LoRA model)
predictor = BiasPredictor()

# Predict bias
result = predictor.predict(
    context="The software engineer was debugging code.",
    sentence="He stayed up all night fixing bugs."
)

print(result)
# {'label': 'stereotype', 'confidence': 0.82, 'label_id': 1}
```

### Evaluate Custom Model

```python
from src.models.logistic_regression import LogisticRegressionClassifier
from src.preprocess import get_train_val_test_split

# Load data with same splits as transformer models
splits = get_train_val_test_split()
test_texts, test_labels = splits["test"]

# Load and evaluate
classifier = LogisticRegressionClassifier.load()
metrics = classifier.evaluate(test_texts, test_labels)
print(metrics)
```

## 📄 License

This project is for educational and demonstration purposes.

---

*Built with a focus on responsible AI development and production-grade engineering practices.*
