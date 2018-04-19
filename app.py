from flask import Flask, render_template
import pickle


app = Flask(__name__)

@app.route('/')
def Prediction():
    tempclf = pickle.load(open('predict.p','rb'))
    x = [5,6]
    return render_template('home.html', x = x)

if __name__ == '__main__':
    app.run(debug=True)
