from flask import Blueprint, render_template, redirect, url_for, flash, session
from flask_login import current_user
from app.models import User
from app import db

profiles = Blueprint('profiles', __name__)

@profiles.route("/profiles")
def admin_dashboard():
    if not (session.get('is_admin') or (current_user.is_authenticated and current_user.role == 'admin')):
        flash("Access denied.", "danger")
        return redirect(url_for('main.home'))

    users = User.query.all()
    return render_template("profiles.html", users=users)

@profiles.route("/delete_user/<int:user_id>")
def delete_user(user_id):
    if not (session.get('is_admin') or (current_user.is_authenticated and current_user.role == 'admin')):
        flash("Unauthorized action.", "danger")
        return redirect(url_for('main.home'))

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash("User deleted.", "success")
    return redirect(url_for('profiles.admin_dashboard'))
