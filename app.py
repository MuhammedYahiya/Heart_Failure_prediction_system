from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from flask import jsonify



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/heart_disease_prediction'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'c34a2b997bf01856ea6f64e6106eb148'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(256))


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    sex = db.Column(db.Integer)
    cp = db.Column(db.Integer)
    trestbps = db.Column(db.Integer)
    chol = db.Column(db.Integer)
    fbs = db.Column(db.Integer)
    restecg = db.Column(db.Integer)
    thalach = db.Column(db.Integer)
    exang = db.Column(db.Integer)
    oldpeak = db.Column(db.Float)
    slope = db.Column(db.Integer)
    ca = db.Column(db.Integer)
    thal = db.Column(db.Integer)
    result = db.Column(db.Integer)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            return redirect(url_for('prediction'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')



@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    thalach = float(request.form['thalach'])
    oldpeak = float(request.form['oldpeak'])

    sex_value = int(request.form['sex_0'])
    cp_value = int(request.form['cp_0'])
    fbs_value = int(request.form['fbs_0'])
    restecg_value = int(request.form['restecg_0'])
    exang_value = int(request.form['exang_0'])
    slope_value = int(request.form['slope_0'])
    ca_value = int(request.form['ca_0'])
    thal_value = int(request.form['thal_1'])

    sex_0 = 1 if sex_value == 0 else 0
    sex_1 = 1 if sex_value == 1 else 0

    cp_0 = 1 if cp_value == 0 else 0
    cp_1 = 1 if cp_value == 1 else 0
    cp_2 = 1 if cp_value == 2 else 0
    cp_3 = 1 if cp_value == 3 else 0

    fbs_0 = 1 if fbs_value == 0 else 0
    fbs_1 = 1 if fbs_value == 1 else 0

    restecg_0 = 1 if restecg_value == 0 else 0
    restecg_1 = 1 if restecg_value == 1 else 0
    restecg_2 = 1 if restecg_value == 2 else 0

    exang_0 = 1 if exang_value == 0 else 0
    exang_1 = 1 if exang_value == 1 else 0

    slope_0 = 1 if slope_value == 0 else 0
    slope_1 = 1 if slope_value == 1 else 0
    slope_2 = 1 if slope_value == 2 else 0

    ca_0 = 1 if ca_value == 0 else 0
    ca_1 = 1 if ca_value == 1 else 0
    ca_2 = 1 if ca_value == 2 else 0

    thal_1 = 1 if thal_value == 1 else 0
    thal_2 = 1 if thal_value == 2 else 0
    thal_3 = 1 if thal_value == 3 else 0

    input_features = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "sex_0": sex_0,
        "sex_1": sex_1,
        "cp_0": cp_0,
        "cp_1": cp_1,
        "cp_2": cp_2,
        "fbs_0": fbs_0,
        "fbs_1": fbs_1,
        "restecg_0": restecg_0,
        "restecg_1": restecg_1,
        "restecg_2": restecg_2,
        "exang_0": exang_0,
        "exang_1": exang_1,
        "slope_0": slope_0,
        "slope_1": slope_1,
        "slope_2": slope_2,
        "ca_0": ca_0,
        "ca_1": ca_1,
        "ca_2": ca_2,
        "thal_1": thal_1,
        "thal_2": thal_2,
        "thal_3": thal_3
    }
    df = pd.DataFrame([input_features])
    output = model.predict(df)

    if output == 1:
        res_val = "heart disease"
    else:
        res_val = "no heart disease"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
