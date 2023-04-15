from wtforms import Form, StringField, BooleanField, PasswordField, validators,  widgets, FileField, SubmitField

class RegistrationForm(Form):
	"""docstring for RegistrationForm"""
	username = StringField('Username', [validators.Length(min=4, max=10)])
	github_username = StringField('Github Username')#, widget=widgets.TextArea())
	email = StringField('Email')    #email vaidation
	password = PasswordField('Password', [validators.InputRequired(), validators.EqualTo('confirm', message="Passwords must match")])
	confirm = PasswordField('Repeat Password')


class LoginForm(Form):
	"""docstring for RegistrationForm"""
	username = StringField('Username', [validators.Length(min=4, max=20)])
	password = PasswordField('Password', [validators.InputRequired()])
