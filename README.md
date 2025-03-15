# Sentiment Analysis Project

This is a sentiment analysis Project built using **Multinomial Naïve Bayes**. It classifies text as Positive or Negative. The API was built using **Flask**.

## How to Run Notebook Locally `Sentiment_analysis_project.ipynb`
1. Install required dependencies: `pip install numpy pandas matplotlib joblib nltk seaborn scikit-learn tensorflow kaggle`
2. The datasets can be gotten from  [Sentiment Dataset](https://drive.google.com/drive/folders/1Vs3IOhzfNxWzqoFqGInVuABOuEPn_MEk?usp=drive_link)
3. At the second cell, under the markdown **Loading the dataset**, insert the link to each csv file and run the code

## How to Run the Flask API `(app.py)`
1.  Install **Flask**: `pip install flask`
2.  Download the trained model & vectorizer:
- Model: `sentiment_analysis_model.pkl`
- Vectorizer: `vectorization_algo.pkl`
3. Insert the link to the model and vectorizer at `path_to_model` and `path_to_vectorizer` respectively
4. Run the code to start the **server**: `python app.py`
5. Use Postman or cURL to test `/analyze_sentiment` endpoint.


_Feel free to modify the codes as you please_

## Future Improvements
- Improve model accuracy (currently at `86.24%`)
- Deploy API online
- Experiment with deep learning models (BERT, LSTMs)
