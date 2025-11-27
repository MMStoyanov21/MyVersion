"""
Module: profiles/units.py
Description: Utility functions for profiles metrics.
"""

from app.models import User, Survey

def count_users():
    """
    Return total count of registered users.
    """
    return User.query.count()

def count_surveys():
    """
    Return total count of survey entries.
    """
    return Survey.query.count()

def get_latest_surveys(limit=10):
    """
    Retrieve the most recent surveys.

    Args:
        limit (int): Number of recent entries to return.
    """
    return Survey.query.order_by(Survey.date_posted.desc()).limit(limit).all()
