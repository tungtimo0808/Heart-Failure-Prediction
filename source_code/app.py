import joblib
import pandas as pd
from flask import Flask, request, render_template

# Load model (đã one-hot trước)
model = joblib.load("xgb_pipeline.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Validate inputs
            age = float(request.form["age"])
            if age < 0 or age > 120:
                return render_template("index.html", result="Error: Age must be between 0 and 120")

            sex = request.form["sex"]
            if sex not in ["M", "F"]:
                return render_template("index.html", result="Error: Sex must be Male or Female")
            sex = 1 if sex == "M" else 0  # Convert to binary (0 for F, 1 for M)

            restingbp = float(request.form["restingbp"])
            if restingbp < 0:
                return render_template("index.html", result="Error: RestingBP must be non-negative")

            cholesterol = float(request.form["cholesterol"])
            if cholesterol < 0:
                return render_template("index.html", result="Error: Cholesterol must be non-negative")

            maxhr = float(request.form["maxhr"])
            if maxhr < 0:
                return render_template("index.html", result="Error: MaxHR must be non-negative")

            oldpeak = float(request.form["oldpeak"])
            fastingbs = int(request.form["fastingbs"])
            if fastingbs not in [0, 1]:
                return render_template("index.html", result="Error: FastingBS must be Yes or No")

            exerciseangina = int(request.form["exerciseangina"])
            if exerciseangina not in [0, 1]:
                return render_template("index.html", result="Error: ExerciseAngina must be Yes or No")

            chestpaintype = request.form["chestpaintype"]
            if chestpaintype not in ["ASY", "NAP", "ATA", "TA"]:
                return render_template("index.html", result="Error: Invalid ChestPainType")

            st_slope = request.form["st_slope"]
            if st_slope not in ["Up", "Flat", "Down"]:
                return render_template("index.html", result="Error: Invalid ST_Slope")

            restingecg = request.form["restingecg"]
            if restingecg not in ["Normal", "LVH", "ST"]:
                return render_template("index.html", result="Error: Invalid RestingECG")

            # One-hot encoding
            raw_input = {
                "Age": age,
                "Sex": sex,  # Now sex is already an integer (0 or 1)
                "RestingBP": restingbp,
                "Cholesterol": cholesterol,
                "FastingBS": fastingbs,
                "MaxHR": maxhr,
                "ExerciseAngina": exerciseangina,
                "Oldpeak": oldpeak,
                "ChestPainType_ASY": 0,
                "ChestPainType_ATA": 0,
                "ChestPainType_NAP": 0,
                "ChestPainType_TA": 0,
                "RestingECG_LVH": 0,
                "RestingECG_Normal": 0,
                "RestingECG_ST": 0,
                "ST_Slope_Down": 0,
                "ST_Slope_Flat": 0,
                "ST_Slope_Up": 0
            }

            raw_input[f"ChestPainType_{chestpaintype}"] = 1
            raw_input[f"ST_Slope_{st_slope}"] = 1
            raw_input[f"RestingECG_{restingecg}"] = 1

            input_data = pd.DataFrame([raw_input])

            # Predict
            proba = model.predict_proba(input_data)[0, 1]
            pred = int(proba >= 0.5)

            result = f"Probability of Heart Disease: {proba:.2%} → Prediction: {'Positive' if pred == 1 else 'Negative'}"
            return render_template("index.html", result=result)

        except ValueError as e:
            return render_template("index.html", result=f"Error: {str(e)}")

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)