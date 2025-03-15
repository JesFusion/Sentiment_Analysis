# Sentiment Analysis Project

This is a sentiment analysis Project built using **Multinomial Naïve Bayes**. It classifies text as Positive or Negative. The API was built using **Flask**.

## How to Run Locally
1. Install required dependencies: `pip install numpy pandas matplotlib joblib nltk seaborn scikit-learn tensorflow kaggle`
2. Start the server: `python app.py`
3. Use Postman or cURL to test `/analyze_sentiment` endpoint.

## Future Improvements
- Improve model accuracy (currently at `87%`)
- Deploy API online
- Experiment with deep learning models (BERT, LSTMs)
