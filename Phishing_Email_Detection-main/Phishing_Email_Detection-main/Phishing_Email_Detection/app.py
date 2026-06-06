from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        email_text = request.form["email"]

        email_vector = vectorizer.transform([email_text])

        prediction = model.predict(email_vector)

        if prediction[0] == 1:
            result = "⚠️ Phishing Email Detected"
        else:
            result = "✅ Safe Email"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)