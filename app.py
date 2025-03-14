from flask import Flask, request, jsonify

import joblib

# initializing flask app

app = Flask(__name__)

# loading the saved model and vectorizer...

naive_bayes_model = joblib.load("https://github.com/JesFusion/Sentiment_Analysis/blob/main/sentiment_analysis_model.pkl")

vectorization_algo = joblib.load("https://github.com/JesFusion/Sentiment_Analysis/blob/main/vectorization_algo.pkl")

@app.route("/", methods = ['GET'])
def home():
    return jsonify({"mesage": "Sentiment Analysis API is running..."})

@app.route("/analyze_sentiment", methods = ["POST"])
def predict():
    try:
        # getting JSON data from request...
        inp_data = request.get_json()
        reviewed_text = inp_data.get("text", "")
    
        # checking if text is provided...
        if not reviewed_text:
            return jsonify({"Input error": "No text was provided!"}), 400
    
        # Transforming text using TF-IDF...
        transformed_text = vectorization_algo.transform([reviewed_text])

        # predicting sentiment...
        model_prediction = naive_bayes_model.predict(transformed_text)[0]

        # converting prediction to readable form
        model_output_map = {0: "Negative Response", 1: "Positive Response"}
        
        model_output_label = model_output_map.get(model_prediction, "Unknown [J]")
    
        return jsonify({"Input text": reviewed_text, "User's sentiment": model_output_label})

    except Exception as e:
        return jsonify({"Syntax error": str(e)}), 500


# running the app
if __name__ == "__main__":
     app.run(debug = True)
