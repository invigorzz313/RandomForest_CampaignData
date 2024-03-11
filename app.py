from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__,template_folder='template')
model = pickle.load(open('finalized_model.pickle', 'rb'))

@app.route("/",methods=["GET"])
def Home_page():
    return render_template("index.html")


standard_to = StandardScaler()
@app.route("/predict",methods=["POST"])
def prediction_page():
    if request.method == "POST":
        sender = float(request.form["sender"])
        subject_len = float(request.form["subject_len"])
        body_len = float(request.form["body_len"])
        mean_paragraph_len = float(request.form["mean_paragraph_len"])
        day_of_week = float(request.form["day_of_week"])
        is_weekend = float(request.form["is_weekend"])
        times_of_day = float(request.form["times_of_day"])
        category = float(request.form["category"])
        product = float(request.form["product"])
        no_of_CTA = float(request.form["no_of_CTA"])
        mean_CTA_len = float(request.form["mean_CTA_len"])
        is_image = float(request.form["is_image"])
        is_personalised = float(request.form["is_personalised"])
        is_quote = float(request.form["is_quote"])
        is_timer = float(request.form["is_timer"])
        is_emoticons = float(request.form["is_emoticons"])
        is_discount = float(request.form["is_discount"])
        is_price = float(request.form["is_price"])
        is_urgency = float(request.form["is_urgency"])
        target_audience = float(request.form["target_audience"])
        

        prediction = model.predict([[ sender,subject_len, body_len, mean_paragraph_len, day_of_week, is_weekend, times_of_day, category, product, no_of_CTA, mean_CTA_len, is_image, is_personalised, is_quote, is_timer, is_emoticons, is_discount, is_price, is_urgency, target_audience]])
        return render_template('result.html',result=prediction)

    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)




