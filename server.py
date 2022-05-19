from this import d
from flask import Flask, redirect, render_template, request, url_for
import numpy as np
import pickle
import pandas as pd
app = Flask(__name__)

model = pickle.load(open("cardio-model.pkl", "rb"))
@app.route('/', methods=['GET', 'POST'])
def cardio():
     if request.method == 'POST':
        age = request.form['age']
        height = request.form['height']
        gender = request.form['gender']
        weight = request.form['weight']
        hbp = request.form['hbp']
        lbp = request.form['lbp']
        cl = request.form['cl']
        glucose = request.form['glucose']
        option2 = request.form['Physical ']
        option = request.form['Smoking ']
        option1 = request.form['Alcohol intake ']

        input_variables = pd.DataFrame([[age,height, gender, weight, hbp, lbp, cl, glucose, option, option1, option2]],
                                       columns=['age','height', 'gender', 'weight', 'hbp', 'lbp', 'cl', 'glucose', 'option', 'option1', 'option2'],
                                       dtype='float', index=['input'])
        output = model.predict(input_variables)[0]
        # print(predict)
        # output = prediction[0]

        condition = ["Without a Cardiovascular disease", "With a Cardiovascular disease"]
        return render_template(
            "index.html", prediction_text="Patient's Condition: {}".format(condition[output])
        )
      #   result = condition[output]
      #   return redirect(url_for("result", result = result ))
     else:
        return render_template('index.html')
# @app.route("/results", methods = ['GET', 'POST'])
# def result():
#     result = request.args.get('result')
#     return render_template('i.html', prediction_text = result)
if __name__ == '__main__':
     app.run(debug = True, port=9000)
