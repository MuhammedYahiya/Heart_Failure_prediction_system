from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/heart_disease_prediction'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'c34a2b997bf01856ea6f64e6106eb148'

db = SQLAlchemy(app)
migrate = Migrate(app, db)
