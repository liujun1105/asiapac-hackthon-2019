from flask_wtf import FlaskForm
from wtforms import StringField


class RegForm(FlaskForm):
    client_name = StringField('Client Name')
    machine_name = StringField('Machine Name')
    username = StringField('Username')

