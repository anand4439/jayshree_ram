from cgi import test
from flask import Flask,request,render_template,redirect,url_for
import pandas as pd
import numpy as np
import pickle

Lin_mod=pickle.load(open('linmodel.pkl','rb'))

app = Flask(__name__)

@app.route("/")
def welcome():
    return render_template("home.html")

@app.route("/prediction",methods=["POST"])
def home():
    AT = request.form['AT']
    EV = request.form['EV']
    AP = request.form['AP']

    arr = np.array([[AT, EV, AP]])
    
    test_df = pd.DataFrame(arr)

    pred = Lin_mod.predict(test_df)

    return render_template('after.html', data=pred[0])


if __name__=="__main__":
    app.run(host='0.0.0.0',port=7000,debug=True)