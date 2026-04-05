# NaijaSenti — Nigerian Twitter Sentiment Analysis

Multilingual sentiment classification across four Nigerian languages: **Hausa, Igbo, 
Nigerian Pidgin, and Yorùbá**. This project fine-tunes and compares four model 
architectures on the NaijaSenti corpus, from classical ML to a pretrained 
African language transformer.

## Live Demo
[Try it on Hugging Face Spaces](https://huggingface.co/spaces/KingsleyElo/naija-sentiment)

> Note: The live demo runs Logistic Regression and LSTM. AfroXLMR (best model, 
> 0.74 F1) is documented in the results table and available as a trained model 
> on [Hugging Face](https://huggingface.co/KingsleyElo/naija-sentiment-models).

---

## Results

| Model               | Overall Macro F1 | Hausa | Igbo | Pidgin | Yoruba |
|---------------------|------------------|-------|------|--------|--------|
| Logistic Regression | 0.69             | 0.71  | 0.73 | 0.44   | 0.68   |
| SimpleRNN           | 0.69             | 0.72  | 0.74 | 0.38   | 0.69   |
| LSTM (V2)           | 0.71             | 0.75  | 0.75 | 0.43   | 0.69   |
| **AfroXLMR**        | **0.74**         | 0.77  | 0.78 | 0.49   | 0.70   |

> **Note on Pidgin neutral:** The neutral class has only 72 training samples in 
> Pidgin, causing consistent underperformance across all models. This is a known 
> data limitation in the NaijaSenti corpus, not a modelling failure.

---

## Setup

**Requirements:** Python 3.10, pipenv
```bash
git clone https://github.com/KingsleyElo/naija-sentiment.git
cd naija-sentiment
pipenv install
pipenv shell
```

Run notebooks in order: `01 → 02 → 03 → 04 → 05`

> AfroXLMR (notebook 05) requires a GPU. Training was done on Google Colab 
> (T4 GPU, ~25 minutes). The notebook is fully runnable locally for inference 
> only if model weights are downloaded from the releases section.

---

## Models

### Logistic Regression
Classical baseline with TF-IDF features. Aggressive text preprocessing: 
lowercasing, URL removal, punctuation, emoji, stopword removal, and lemmatization.

### SimpleRNN
Keras sequential RNN with embedding layer. Minimal preprocessing (URLs and 
emojis only) to preserve sequence structure. Matched the LogReg baseline at 
0.69 overall but underperformed on Pidgin — LSTM's gated memory was necessary 
for consistent gains across all languages.

### LSTM (V2)
LSTM with Dropout, L2 regularisation, and EarlyStopping. 
Tokenizer shared with SimpleRNN. Outperformed LogReg baseline on all languages.

### AfroXLMR
Fine-tuned [Davlan/afro-xlmr-base](https://huggingface.co/Davlan/afro-xlmr-base) 
(270M parameters) using PyTorch. AdamW optimizer, linear warmup scheduler, 
gradient clipping, and class-weighted CrossEntropyLoss. Best overall performance.

---

## Key Findings

- AfroXLMR's pretrained African language knowledge delivers consistent gains 
  over classical and recurrent baselines across Hausa, Igbo, and Yoruba
- Yoruba diacritics are handled natively by the XLM-R tokenizer — an advantage 
  over the Keras tokenizer used in earlier models
- Pidgin neutral underperformance (F1: 0.07 with AfroXLMR) is a data problem: 
  72 training samples cannot support any reliable classifier
- SimpleRNN matched LogReg overall (0.69) but underperformed on Pidgin,
  LSTM's gated memory is necessary for consistent gains across all languages
- Random baseline for a 3-class problem = 0.33. AfroXLMR achieves 2.2× that

---

## Dataset

This project uses the **NaijaSenti** corpus — a Nigerian Twitter sentiment 
dataset covering Hausa, Igbo, Nigerian Pidgin, and Yorùbá.

**Dataset splits used (after leakage removal):**

| Split | Size   |
|-------|--------|
| Train | 36,569 |
| Val   | 7,599  |
| Test  | 17,654 |

If you use this dataset in your work, please cite the original authors:
```bibtex
@misc{muhammad2022naijasenti,
      title={NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis}, 
      author={Shamsuddeen Hassan Muhammad and David Ifeoluwa Adelani and Sebastian Ruder 
              and Ibrahim Said Ahmad and Idris Abdulmumin and Bello Shehu Bello 
              and Monojit Choudhury and Chris Chinenye Emezue and Saheed Salahudeen Abdullahi 
              and Anuoluwapo Aremu and Alipio Jeorge and Pavel Brazdil},
      year={2022},
      eprint={2201.08277},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

## Author

**Kingsley Eloebhose**  
ML Engineer  
[GitHub](https://github.com/KingsleyElo) · [LinkedIn](https://www.linkedin.com/in/kingsley-eloebhose-77ab41379) · [Portfolio](https://kingsleyelo.github.io)
