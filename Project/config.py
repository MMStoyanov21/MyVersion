import os

class Config:
    SECRET_KEY = 'your_secret_key_here'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'  # или път към друга база
    SQLALCHEMY_TRACK_MODIFICATIONS = False
