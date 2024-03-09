from flask.cli import FlaskGroup
from app import app, db  # Assuming your Flask application instance is named 'app'

cli = FlaskGroup(app)

if __name__ == "__main__":
    cli()
