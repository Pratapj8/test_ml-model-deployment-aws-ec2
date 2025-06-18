from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')

    error = None
    if not cgpa or not iq or not profile_score:
        error = "All fields are required."
        return render_template('index.html', result=error)

    try:
        cgpa = float(cgpa)
        iq = int(iq)
        profile_score = int(profile_score)
    except ValueError:
        error = "Invalid input. Please enter valid numbers."
        return render_template('index.html', result=error)

    # prediction
    result = model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))

    if result[0] == 1:
        result = 'placed'
    else:
        result = 'not placed'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)