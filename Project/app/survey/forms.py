from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

class UploadCSVForm(FlaskForm):
    csv_file = FileField('Upload CSV File', validators=[
        FileRequired(message="CSV file is required."),
        FileAllowed(['csv'], 'CSV files only!')
    ])
    submit = SubmitField('Estimate From File')
