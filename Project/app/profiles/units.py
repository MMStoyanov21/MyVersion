from app.models import User, Survey

def count_users():

    return User.query.count()

def count_surveys():

    return Survey.query.count()

def get_latest_surveys(limit=10):

    return Survey.query.order_by(Survey.date_posted.desc()).limit(limit).all()
