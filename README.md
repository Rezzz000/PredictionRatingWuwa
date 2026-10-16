# Wuthering Waves Review Rating Prediction

Predicting 1–5 star ratings from the text of **Wuthering Waves** reviews on Google Play Store, using classic NLP pipelines, machine learning, and deep learning.

The goal is a model that game developers can use to estimate the sentiment/rating behind player feedback from review-style text.

## Highlights

- **Data**: ~2,000 English reviews scraped from Google Play (`com.kurogame.wutheringwaves.global`)
- **Task**: 5-class classification (rating 1–5)
- **Text representations**: TF-IDF (unigrams + bigrams) and Word2Vec (Skip-Gram)
- **Models**: Random Forest, Linear SVM, Bidirectional LSTM
- **Imbalance handling**: SMOTE and class weighting, compared against the original data

## Pipeline

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Scraping | Collect reviews with `google-play-scraper` (newest + most relevant sorting), deduplicate, save to CSV |
| 2 | EDA | Rating distribution, per-rating word clouds, top words, slang detection, review length analysis |
| 3 | Preprocessing | Case folding, URL/number/punctuation removal, slang normalization, tokenization, stopword removal, lemmatization |
| 4 | Text representation | TF-IDF (5,000 features, 1–2 grams) and Word2Vec (100-dim, Skip-Gram, averaged per document) |
| 5 | Machine learning | Random Forest (`RandomizedSearchCV`) and LinearSVC (`GridSearchCV`) on both representations |
| 6 | Deep learning | Bidirectional LSTM on Word2Vec document vectors, with a small hyperparameter search and early stopping |
| 7 | Evaluation | Accuracy, weighted Precision/Recall/F1, classification reports, confusion matrices |
| 8 | Imbalance handling | SMOTE on TF-IDF; class weights on Word2Vec and the BiLSTM; before/after comparison |

## Experiments

Each model is trained on an 80/20 stratified train/test split (`random_state=42`).

| Algorithm | Representation | Tuning |
|-----------|----------------|--------|
| Random Forest | TF-IDF, Word2Vec | `n_estimators`, `max_depth`, `min_samples_split` (RandomizedSearchCV, 3-fold, F1-weighted) |
| LinearSVC | TF-IDF, Word2Vec | `C`, `max_iter` (GridSearchCV, 3-fold, F1-weighted) |
| BiLSTM | Word2Vec | LSTM units, dropout rate, learning rate (3 configs, best by validation accuracy) |

Imbalance handling is then applied and all models are re-evaluated:

- **SMOTE** for the TF-IDF models
- **Class weight (`balanced`)** for the Word2Vec models and the BiLSTM

## Results

Run the notebook to reproduce results. Scores depend on the reviews available at scraping time, so they will vary between runs.

The notebook generates:

- `hasil_evaluasi_lengkap.csv`: metrics for every model and data condition (original vs. balanced)
- `confusion_matrix_semua_model.png`, `perbandingan_f1_score.png`, `perbandingan_imbalance_handling.png`: evaluation plots

You can paste your own results here once you have them:

| Model | Representation | Data condition | Accuracy | Precision | Recall | F1 (weighted) |
|-------|----------------|----------------|----------|-----------|--------|---------------|
| _fill in_ | | | | | | |

## Getting Started

### Requirements

- Python 3.9+
- Jupyter Notebook, JupyterLab, or Google Colab

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install google-play-scraper gensim imbalanced-learn wordcloud nltk \
            scikit-learn tensorflow pandas numpy matplotlib seaborn
```

### Run

```bash
jupyter notebook wuthering_waves_nlp.ipynb
```

Run the cells top to bottom. The first cells download the required NLTK resources (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`).

> **Note:** Scraping hits Google Play live, so it needs an internet connection and may take a few minutes. A 1-second delay between requests is built in to avoid being blocked.

## Project Structure

```
.
├── wuthering_waves_nlp.ipynb        # Full pipeline: scraping → modeling → evaluation
├── wuthering_waves_reviews.csv      # Raw scraped data (generated)
├── wuthering_waves_preprocessed.csv # Cleaned text (generated)
├── word2vec_model.bin               # Trained Word2Vec model (generated)
├── hasil_evaluasi_lengkap.csv       # Final evaluation table (generated)
└── *.png                            # EDA and evaluation plots (generated)
```

## Tech Stack

`google-play-scraper` · `pandas` · `NumPy` · `NLTK` · `scikit-learn` · `gensim` · `imbalanced-learn` · `TensorFlow/Keras` · `matplotlib` · `seaborn` · `wordcloud`

## Limitations and Ideas for Improvement

- **Small dataset**: ~2,000 reviews is limited for 5-class prediction, especially for the neighboring classes (2–4 stars), which are hard to separate from text alone.
- **Imbalanced classes**: 5-star (and 1-star) reviews typically dominate on app stores; check the EDA output for the exact distribution.
- **BiLSTM input**: the LSTM receives each averaged Word2Vec document vector as a single time step, so it does not model word order. Feeding token sequences through an embedding layer would let it use the sequential structure.
- **SMOTE on text**: SMOTE interpolates in TF-IDF space, so the synthetic samples are not real text.
- **Possible next steps**: pre-trained embeddings (GloVe, fastText), transformer models (BERT/DistilBERT), collapsing ratings into negative/neutral/positive, and evaluating on a held-out time period.


## Acknowledgements

- Review data from Google Play Store via [`google-play-scraper`](https://github.com/JoMingyu/google-play-scraper)
- Wuthering Waves is a game by Kuro Games; this project is an independent, educational analysis and is not affiliated with Kuro Games
