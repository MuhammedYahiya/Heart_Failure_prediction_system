from extensions import app, db
from flask import render_template, redirect, url_for, request, jsonify
from flask_wtf import FlaskForm
import numpy as np
import pandas as pd
import pickle
from flask_bootstrap import Bootstrap
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import torch
from chat_bot_model.model import NeuralNet
from chat_bot_model.nltk_utils import bag_of_words, tokenize
import random
import json
 

# Load the model
with open('./model/knn.pkl', 'rb') as file:
    model_knn = pickle.load(file)

bootstrap = Bootstrap(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(256))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('Remember me')

class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))
    return render_template("login.html", form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")

@app.route("/disindex")

def disindex():
    return render_template("disindex.html")

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_data = {feature: [request.form[feature]] for feature in features}

        # Convert to DataFrame to preserve feature names
        new_data = pd.DataFrame(input_data)

        # Convert numeric values
        new_data = new_data.astype(float)

        # Make prediction using the loaded model
        prediction = model_knn.predict(new_data)
        
        # Calculate probability of each class
        probabilities = model_knn.predict_proba(new_data)[0]
        
        # Calculate the intensity percentage
        intensity_percentage = probabilities[1] * 100

        if prediction == 1:
            res_val = "Oops! You have Chances of Heart Disease."
        else:
            res_val = "Great! You DON'T have chances of Heart Disease."

        # Determine sleep cycle based on intensity percentage
        sleep_cycle = "9 hours" if intensity_percentage >= 80 else \
                      "8 hours" if intensity_percentage >= 60 else \
                      "7 hours" if intensity_percentage >= 40 else \
                      "6 hours" if intensity_percentage >= 20 else "5 hours"

        return render_template('result.html', prediction_text=res_val, intensity_percentage=intensity_percentage, sleep_cycle=sleep_cycle)


# Load model and data
data = torch.load("./chat_bot_model/data.pth")
model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
with open('./chat_bot_model/intents.json', 'r') as file:
    intents = json.load(file)

# Initialize and load model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

def chatbot_response(sentence):
    # Process input sentence
    sentence = tokenize(sentence)  # Assuming you have a tokenize function
    X = bag_of_words(sentence, all_words)  # Assuming you have a bag_of_words function
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # Predict tag and handle untrained data
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    confidence_threshold = 0.7  # Adjust as needed
    confidence = torch.softmax(output, dim=1)[0, predicted.item()]

    if confidence < confidence_threshold:
        return "That's an interesting question! I'm still learning, and I'm not quite familiar with this topic yet. Would you like me to try finding some information online, or would you prefer to ask something else?"
    else:
        # Find appropriate response
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                return response

@app.route('/chatbot')
def chat():
    return render_template('chat.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        message = data['message']
        response = chatbot_response(message)
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": "An error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True)