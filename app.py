from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    transformed_text = vectorizer.transform([user_text])
    prediction = model.predict(transformed_text)[0]
    
    sentiment = "Positive" if prediction == 1.0 else "Neutral" if prediction == 0.0 else "Negative"
    color_class = "positive" if prediction == 1.0 else "neutral" if prediction == 0.0 else "negative"
    
    return render_template('analyze.html', prediction_text=f'Sentiment: {sentiment}', sentiment_class=color_class)

if __name__ == '__main__':
    app.run(debug=True)
