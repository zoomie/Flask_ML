from flask import Flask, render_template
import dill as pickle
import pandas as pd
from Machine_Learning import Predict_df


app = Flask(__name__)

@app.route('/')
def Prediction():
    df_test = pd.read_csv('test.csv')
    x = Predict_df(df_test)
    return render_template('home.html', x = x)

if __name__ == '__main__':
    app.run(debug=True)
