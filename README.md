# Sentiment Analysis on Social Media Data

This project involves building supervised machine learning models for sentiment classification on social media datasets collected from Twitter and Reddit. The goal is to develop a practical sentiment analysis system that can reliably classify text into positive, neutral, or negative sentiment classes.

---

## Project Overview

- **Datasets:** Two large-scale datasets consisting of 150K+ posts/comments from Twitter and Reddit.
- **Data Exploration:** Analyzed class distribution and text characteristics to understand dataset properties.
- **Preprocessing:** Cleaned and normalized text by removing URLs, mentions, hashtags, punctuation, and stopwords.
- **Feature Engineering:** Converted cleaned text into numerical features using TF-IDF vectorization with unigrams and bigrams.

---

## Models Implemented

- Logistic Regression
- Naive Bayes (Multinomial)
- Support Vector Machine (LinearSVC)
- Random Forest Classifier

Each model was trained and evaluated on Twitter and Reddit datasets independently.

---

## Key Results

- **Best Performance:** Linear Support Vector Machine (SVM) achieved the highest accuracy on Twitter data with **89.28%**.
- Logistic Regression achieved *88.19%* accuracy.
- Random Forest reached *86.09%* accuracy.
- Naive Bayes lagged behind at *73.65%* accuracy due to word independence assumptions.
- Neutral sentiment cases presented challenges, highlighting areas for future improvement.
  
---

## Deployment

- Developed a **Flask web application** serving as an interactive platform where users can input text or upload files to get live sentiment predictions.
- The app includes a professional-looking multi-page interface with navigation menus, styled via CSS.
- The trained model and TF-IDF vectorizer are persisted and loaded to serve prediction requests on arbitrary unseen textual data.

---

## What Makes Our Model Special

- Thorough pipeline from raw noisy social media text through advanced cleaning and n-gram feature extraction.
- Robust experimental comparison of multiple supervised ML algorithms with detailed metric evaluation.
- Real-world readiness demonstrated by deploying the model as a user-friendly interactive web service.
- Balanced handling of a broad range of sentiment expressions across two major social platforms (Twitter and Reddit).
- Clear avenues for extending the model using deep learning for nuanced sentiment detection and scaling to enterprise use cases.

---

## Future Work

- Fine-tune Transformer models (BERT, RoBERTa) to address subtle and neutral sentiments.
- Implement batch processing via file upload to support large-scale sentiment analysis.
- Add user profiles, authentication, and personalized analytics dashboards.
- Deploy on cloud platforms for scalable public access.

---

## Usage Instructions

1. Run `app.py` to start the Flask server.
2. Navigate to `http://127.0.0.1:5000/` to access the web interface.
3. Input text on the "Analyze Text" page or upload CSV files for batch sentiment prediction.
4. View live sentiment results (positive, neutral, negative) with confidence.

---

This project demonstrates both academic rigor and practical engineering skills essential for building scalable sentiment analysis solutions with modern ML techniques.

---

