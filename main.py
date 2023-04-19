import pandas as pd
from flask import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tkinter import ttk
import pandas_profiling as pp
from tkinter import *
import matplotlib 
matplotlib.use('Agg') 

app = Flask(__name__)

global df
global selected_columns
global target_variable
global X_train
global X_test
global Y_test
global y_train_predict
global Y_train
global y_test_predict
global text_col
global null_count



@app.route('/')  
def index():  
    return render_template("index.html")
  
@app.route('/', methods = ['POST'])
def uploadFile():
    global text_col
    global df
    global selected_columns
    file = request.files['file']
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        return "Invalid file format. Please upload a CSV or Excel file."
    selected_columns = []
    text_col = []
    for i in df.columns:
        if df[i].dtype == 'object':
            text_col.append(i)
    if len(text_col)>1:
        return render_template('page1.html', text_col=text_col)    

    return render_template('page1.html')

  
@app.route('/preview')
def previewData():
    global df
    data_head =df.head().to_html(justify='left')
    n_rows = df.shape[0]
    n_cols = df.shape[1]
    return render_template('page1.html', data_head = data_head, n_rows=n_rows, n_cols=n_cols)


@app.route('/drop_col', methods=['POST'])
def drop_column():
    column_name = request.form['column_name']
    global df  # Use the global DataFrame
    # global up_df
    df = df.drop(columns=column_name)
    return render_template('index2.html', df=df.head().to_html(justify='left'))


@app.route('/info')
def info():
    global df 
    global null_count
    
    data_types = df.dtypes.reset_index()
    data_types.columns = ['Column Name', 'Data Type']
    non_null_count = df.count().reset_index()
    non_null_count.columns = ['Column Name', 'Non-Null Count']
    null_count = df.isnull().sum().reset_index()
    null_count.columns = ['Column Name', 'Null Count']
    data_types = pd.merge(data_types, non_null_count, on='Column Name')
    data_types = pd.merge(data_types, null_count, on='Column Name')
    
    
    d_types = df.dtypes.unique()
    df_types = pd.DataFrame({"Data types":d_types}).to_html()
    df_new = pd.DataFrame({'Total Rows': [len(df)], 'Total Columns': [len(df.columns)]})
    
    return render_template('page1.html', data=data_types.to_html(index=False, justify='left'), total=df_new.to_html(index=False, justify='left'),d_types=d_types)


  



if __name__ == '__main__':  
    app.run(debug=True)