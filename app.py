from flask import Flask, render_template, flash, request, url_for, redirect, session
# from flask_admin import Admin
from Models._user import User, db, connect_to_db
from static.Forms.forms import RegistrationForm, LoginForm
from passlib.hash import sha256_crypt
import os.path
import csv
import gc, os
from functools import wraps
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


app = Flask(__name__)
conn = 'sqlite:///'+ os.path.abspath(os.getcwd())+"/data/DataBases/test.db"
dt_model = pickle.load(open('Models/dt_Model_pipeline.pkl', 'rb'))
connect_to_db(app,conn)

@app.route('/register', methods=['GET','POST'])
def register():
    try:
        form = RegistrationForm(request.form)
        if request.method == 'POST':
            _username = request.form['username']
            _email = request.form['email']
            _password = sha256_crypt.encrypt(str(form.password.data))
            user = User(username = _username, email = _email, password = _password)
            db.create_all()
            if User.query.filter_by(username=_username).first() is not None:
                flash('User Already registered with username {}'.format(User.query.filter_by(username=_username).first().username), "warning")
                return render_template('register.html', form=form)
            if User.query.filter_by(email=_email).first() is not None:
                flash('Email is already registered with us {}'.format(User.query.filter_by(email=_email).first().username), "warning")
                return render_template('register.html', form=form)
            flash("Thank you for registering!", "success")
            db.session.add(user)
            db.session.commit()
            db.session.close()
            gc.collect()
            session['logged_in'] = True
            session['username'] = _username
            session.modified = True
            return redirect(url_for('dashboard'))
        return render_template('register.html', form=form)
    except Exception as e:
        return render_template('error.html',e=e)

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args,**kwargs)
        else:
            flash('You need to login first!', "warning")
            return redirect(url_for('login_page'))
    return wrap

def already_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            flash("You are already logged in!", "success")
            return redirect(url_for('dashboard'))
        else:
            return f(*args, **kwargs)
    return wrap

def verify(_username, _password):
    if User.query.filter_by(username=_username).first() is None:
        flash("No such user found with this username", "warning")
        return False
    if not sha256_crypt.verify(_password, User.query.filter_by(username=_username).first().password):
        flash("Invalid Credentials, password isn't correct!", "danger")
        return False
    return True

@app.route('/login', methods=['GET','POST'])
# @already_logged_in
def login():
    try:

        form = LoginForm(request.form)
        if request.method == 'POST':

            _username = request.form['username']
            _password = request.form['password']


            if verify(_username, _password) is False:   
                return render_template('login.html', form=form)
            session['logged_in'] = True
            session['username'] = _username
            gc.collect()
            return redirect(url_for('dashboard'))

        return render_template('login.html')


    except Exception as e:
        return render_template('error.html',e=e)


@app.route('/logout/')
@login_required
def logout():
    session.clear()
    gc.collect()
    flash("You have been logged out!", "success")
    return redirect(url_for('login_page'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predicted_table')
def predicted_table():
    return render_template('predicted_table.html')

@app.route('/prediction_table', methods=['GET', 'POST'])
def prediction_table():
    if request.method == 'POST':
        received_value = pd.DataFrame({
            "CellName": [request.form['CellName']],
            "Time" :    [request.form['Time']],
            "PRBUsageUL" : [ float(request.form['PRBUsageUL'])],
            "PRBUsageDL" : [ float(request.form['PRBUsageDL'])],
            "meanThr_DL" : [ float(request.form['meanThr_DL'])],
            "meanThr_UL" : [ float(request.form['meanThr_UL'])],
            "maxThr_DL" : [float(request.form['maxThr_DL'])],
            "maxThr_UL" : [float(request.form['maxThr_UL'])],
            "meanUE_DL" : [float(request.form['meanUE_DL'])],
            "meanUE_UL" : [float(request.form['meanUE_UL'])],
            "maxUE_DL" : [float(request.form['maxUE_DL'])],
            "maxUE_UL" : [float(request.form['maxUE_UL'])],
            "maxUE_UL+DL" : [float(request.form['maxUE_UL+DL'])]
            }) 
            
        predictions = dt_model.predict(received_value) 
        
        
        is_unsual = []
        
        for predictions in predictions:
            if predictions == 0:
                is_unsual.append(False)  #
            else:
                is_unsual.append(True)  # fraud
        context = {
            'received_value':received_value,
            'predictions': predictions,
            'is_unsual' : is_unsual
            }
        return render_template('predicted_table.html', **context)

    return render_template('prediction_table.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    folder_path = os.path.join(app.root_path, 'data/Uploaded_Here')
    files = os.listdir(folder_path)
    context = {
        'files': files,
        'enumerate': enumerate
    }
    return render_template('dashboard.html', **context)  

app.secret_key = "your_secret_key"
required_columns = ['Time',	'CellName',	'PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL', 'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL']

@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            if file.filename.endswith('.csv'):
                filename = file.filename
                save_location = os.path.join('data/Uploaded_Here', filename)
                file.save(save_location)
                session['filename'] = filename
                
                # Check if the file has the required columns
                try:
                    with open(save_location, 'r') as f:
                        reader = csv.DictReader(f)
                        columns = reader.fieldnames
                        if not all(column in columns for column in required_columns):
                            os.remove(save_location)
                            return render_template('error.html', e ="The uploaded file doesn't have the required columns")
                except Exception as e:
                    os.remove(save_location)
                    return render_template('error.html', e ="The uploaded file doesn't have the required columns")
                
                return redirect(url_for('dashboard'))
            else:
                return render_template('error.html', e ="The uploaded file doesn't have the required columns")
        
    return render_template('upload_csv.html')

@app.errorhandler(404)
def page_not_found(e):
	return render_template('error.html', e=e)
  

@app.route('/show_csv/<filename>', methods=['GET', 'POST'])
def show_csv(filename):
    if not filename:
        flash('Please upload a file first!')
        return redirect(url_for('upload'))

    filepath = os.path.join('data/Uploaded_Here',filename)
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [column for column in reader]
        row = [row for row in reader]
        context= {
            'column':column, 
            'row':row,
            'filename':filename,
            'enumerate': enumerate
        }
    return render_template('show_csv.html', **context)

@app.route('/show_pred_csv/<filename>', methods=['GET', 'POST'])
def show_pred_csv(filename):
    if not filename:
        flash('Please upload a file first!')
        return redirect(url_for('upload'))
    
    filepath = os.path.join('data/Uploaded_Here',filename)
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [column for column in reader]
        row = [row for row in reader]
    df = pd.read_csv(filepath)
    df.isnull().sum()
    df.isnull().shape[0]
    
    predictions = dt_model.predict(df) 
        
    is_unsual = []
    
    for predictions in predictions:
        if predictions == 0:
            is_unsual.append(False)  #
        else:
            is_unsual.append(True)  # fraud
            
    
    context = {
        'column':column, 
        'row':row,
        'filename':filename,
        'enumerate': enumerate,
        'is_unsual' : is_unsual
        }
    return render_template('show_pred_csv.html', **context)

@app.route('/analysis/<filename>', methods=['GET', 'POST'])
def analysis(filename):
    if not filename:
        flash('Please upload a file first!')
        return redirect(url_for('upload'))
    
    filepath = os.path.join('data/Uploaded_Here', filename)
    df = pd.read_csv(filepath)
    df_size = df.shape
    df.isnull().sum()
    df.isnull().shape[0]
    
    predictions = dt_model.predict(df) 
    
    # Cellnames and defining it's graph
    # convert type of the predictions into dataframe
    pred_df = pd.DataFrame(predictions, columns = ['Prediction'])
    # print predict value in the formof data frame
    # print(pred_df)
    
    # conacte prediction values to the original dataframe
    predicted_dataframe = pd.concat([pred_df, df], axis = 'columns')
    # print predicted dataframe with precitions values
    # print(predicted_dataframe.head())
    
    # print all cellnames that in the dataset
    # print(predicted_dataframe.groupby('CellName')['CellName'].agg('count'))
    
    # print only that dataset which has unusual value is 1 
    # cellnames_count = predicted_dataframe[predicted_dataframe['Prediction'] == 1].groupby('CellName')['Prediction'].count()
    # print(cellnames_count)
    
    # group rows by cellname and count the occurrences of 0 in the prediction column
    # cell_count = predicted_dataframe[predicted_dataframe['Prediction'] == 1].groupby('CellName')['Prediction'].count()
    
    # =========================================================================
    # plotting for cellnames
    # group rows by cellname and count the occurrences of 0 and 1 in the prediction column
    cell_count = predicted_dataframe.groupby(['CellName', 'Prediction'])['Prediction'].count().unstack().fillna(0)
    
    # prepare the data for the chart
    cellnames_labels = cell_count.index.tolist()
    cellnames_data0 = cell_count[0].tolist()
    cellnames_data1 = cell_count[1].tolist()
    
    # =========================================================================
    # for time 
   
    # group rows by cellname and count the occurrences of 0 and 1 in the prediction column
    time_count = predicted_dataframe.groupby(['Time', 'Prediction'])['Prediction'].count().unstack().fillna(0)
    
    # prepare the data for the chart
    time_labels = time_count.index.tolist()
    time_data0 = time_count[0].tolist()
    time_data1 = time_count[1].tolist()



    
    # =========================================================================
    
    is_unusual = [prediction == 1 for prediction in predictions]
    df['is_unusual'] = is_unusual
    df['prediction'] = predictions
    df.loc[~df['is_unusual'], 'prediction'] = 0
    
    y = df['is_unusual']
    labels = ["Usual", "Unusual"]
    values = y.value_counts().tolist()  
    return render_template('analysis.html', values = values, labels = labels, df_size = df_size, 
                           cellnames_labels = cellnames_labels, cellnames_data0 = cellnames_data0, 
                           cellnames_data1 = cellnames_data1, time_labels=time_labels,
                           time_data0=time_data0,time_data1=time_data1)


if __name__ == "__main__":
    app.run()
