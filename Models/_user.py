from flask_sqlalchemy import SQLAlchemy
from datetime import datetime 

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    
    id = db.Column(db.Integer,  autoincrement=True, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(120), nullable=False)
    registeron = db.Column(db.DateTime, nullable=False ,default=datetime.now)
    

def __repr__(self):
        return '<User {}>'.format(self.username)


# class Feedback(db.Model):
#     """ This contains feedbacks """

#     __tablename__ = "feedbacks"
#     id = db.Column(db.Integer, autoincrement=True, primary_key=True)
#     feedback_name = db.Column(db.String(80), nullable=False)
#     feedback_email = db.Column(db.String(120), nullable=False)
#     feedback_subject = db.Column(db.String(80), nullable=False)
#     feedback_message = db.Column(db.String(200), nullable=False)
#     feedback_datetime = db.Column(db.DateTime, nullable=False ,default=datetime.now)
    
#     def __repr__(self):
#         return '<Feedback {}>'.format(self.id)
    
    
def connect_to_db(app, spent_database):
    """ Connect the database to our Flask app. """

    # Configure to use the database
    app.config['SQLALCHEMY_DATABASE_URI'] = spent_database
    app.config['SQLALCHEMY_ECHO'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    db.app = app
    db.init_app(app)
    
    