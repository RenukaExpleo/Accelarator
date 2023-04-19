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

@app.route('/visualization')
def visualization():
 return render_template('visualization.html')


@app.route('/boxplot')
def boxplot():
    global df
    df=df

    num_rows = (len(df.columns) + 2) // 3  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(10,15))
    extra_graphs = num_rows * 3 - len(df.columns)
    # Create a boxplot for each column in the DataFrame
    for i, col in enumerate(df.columns):
        sns.boxplot(data=df, x=col, ax=axes[i // 3, i % 3])

    for j in range(1, extra_graphs+1):
        fig.delaxes(axes[num_rows-1][-j])
        
    plt.suptitle('Boxplot For Data Frame')
    file = "static/img11.png"
    plt.savefig(file) 
    # return the image file 
    plt.tight_layout()    
    return render_template('visualization.html', box_url = file)

@app.route('/histogram')
def histogram():
    global df
    df=df

    num_rows = (len(df.columns) + 2) // 3  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(10,15))
    extra_graphs = num_rows * 3 - len(df.columns)
    # Create a boxplot for each column in the DataFrame
    for i, col in enumerate(df.columns):
        sns.histplot(data=df, x=col, ax=axes[i // 3, i % 3])
    
    for j in range(1, extra_graphs+1):
        fig.delaxes(axes[num_rows-1][-j])

    plt.suptitle('Histogram For Data Frame')
    file = "static/img4.png"
    plt.savefig(file) 
    # return the image file 
    plt.tight_layout()    
    return render_template('visualization.html', histplot_url = file)


@app.route('/back')
def back():

    return render_template('page1.html')

@app.route('/profiling')
def profiling():
    global df
    profile = pp.ProfileReport(df)
    
    # get the summary statistics using skim()
    summary = profile.to_html()
    return render_template('profiling.html', data_head = summary)

@app.route('/imputation')
def imputation():

    return render_template('imputation.html')

@app.route('/pairplot')
def pairplot():
    global df
    df=df
    sns.pairplot(df)
    plt.suptitle('Pairplot For Data Frame')
    file = "static/img4.png"
    plt.savefig(file) 
    # return the image file 
    plt.tight_layout() 
    return render_template('visualization.html', pairplot_url = file)

@app.route('/show_nullSum')
def showNullSum():
    global df

    nullSum=df.isnull().sum()
    nullSum_df_html =nullSum.to_frame(name='Null_values_per_column').to_html(justify='left')
    return render_template('imputation.html', data_nullSum = nullSum_df_html)

@app.route('/null_plot')
def nullplot():
    global df
    data_types = df.dtypes.reset_index()
    data_types.columns = ['Column Name', 'Data Type']
    non_null_count = df.count().reset_index()
    non_null_count.columns = ['Column Name', 'Non-Null Count']
    null_count = df.isnull().sum().reset_index()
    null_count.columns = ['Column Name', 'Null Count']
    data_types = pd.merge(data_types, non_null_count, on='Column Name')
    data_types = pd.merge(data_types, null_count, on='Column Name')
    # data_types = data_types
    # null_count = df.isnull().sum()
    barwidth = 0.25
    fig, ax = plt.subplots(figsize=(10,5))
    bar1 = np.arange(len(data_types))
    bar2 = [x+barwidth for x in bar1]


    plt.bar(bar1, data_types['Null Count'].tolist(), width=barwidth, label='NUll', color='c')
    plt.bar(bar2, data_types['Non-Null Count'].tolist(), width=barwidth, label='Non - Null', color='m')


    for bars in ax.containers:
         ax.bar_label(bars)
    plt.xlabel('Columns')
    plt.ylabel('Number of Null Values')
    plt.xticks([r + barwidth for r in range(len(data_types))],data_types['Column Name'].tolist())
    plt.legend()
    # plt.show()
    file = "static/img10.png"
    plt.savefig(file)
    return render_template('imputation.html', nullCount_plot_url = file, data_types=data_types)


@app.route('/nullhistogram')
def hist():
    global text_col
    global df
    null_dict = df.isnull().sum().to_dict()
    null_cols = []
    for key, value in null_dict.items():
        if value != 0:
            null_cols.append(key)
    try:
        plt.figure(figsize=(10,5))
        df[null_cols].hist()
    except:
        return "Either there are no columns with null values or they have been removed."
    plt.suptitle('Histograms of columns having null values')
    plt.tight_layout()
    plt_file = "static/img.png"
    plt.savefig(plt_file)
    return render_template('imputation.html', plot_url = plt_file, text_col=text_col)

@app.route('/nullbox')
def nullbox():
    global text_col
    global df
    null_dict = df.isnull().sum().to_dict()
    null_cols = []
    for key, value in null_dict.items():
        if value != 0:
            null_cols.append(key)
    try:
        plt.figure(figsize=(10,5))
        df[null_cols].boxplot(figsize=(10,5))
    except:
        return "Either there are no columns with null values or they have been removed."
    plt.suptitle('Box Plot For columns having null values')
    plt.tight_layout()
    file = "static/nullbox.png"
    plt.savefig(file)
    return render_template('imputation.html', nullbox_url = file, text_col=text_col)


 
@app.route('/fill_null_mean')
def fillNull():
    global df
    if df.isnull().values.any():
        df.fillna(df.mean(), inplace=True)
        nullSum=df.isnull().sum()
        null_df_html =nullSum.to_frame(name='Null_values_per_column').to_html()
        return render_template('imputation.html', data_nullSum = null_df_html)
    else:
        return "Either there are no columns with null values or they have been removed."

@app.route('/fill_null_median')
def median():
    global df
    if df.isnull().values.any():
        df.fillna(df.median(), inplace=True)
        nullSum=df.isnull().sum()
        null_df_html =nullSum.to_frame(name='Null_values_per_column').to_html()
        return render_template('imputation.html', data_nullSum = null_df_html)
    else:
        return "Either there are no columns with null values or they have been removed."

@app.route('/fill_null_mode')
def mode():
    global df
    if df.isnull().values.any():
        df.fillna(df.mode(), inplace=True)
        nullSum=df.isnull().sum()
        null_df_html =nullSum.to_frame(name='Null_values_per_column').to_html()
        return render_template('imputation.html', data_nullSum = null_df_html)
    else:
        return "Either there are no columns with null values or they have been removed."
    
@app.route('/fillna_auto')
def autonull():
    global text_col
    try:
        null_dict = df.isnull().sum().to_dict()
        for key, value in null_dict.items():
            if value != 0:
                skewness = df[key].skew()
                if df[key].nunique() <= 10 and df[key].nunique() / len(df[key]) <= 0.5:
                    print(f"{key} is categorical and mode is {df[key].mode()[0]}")
                    df[key].fillna(df[key].mode()[0], inplace=True)

                elif abs(skewness) > 1:
                    print(f"{key} is highly skewed ({skewness:.2f}), fill with median")
                    df[key].fillna(df[key].median(), inplace=True)
                    
                else:
                    print(f"{key} is normally distributed ({skewness:.2f}), fill with mean")
                    df[key].fillna(df[key].mean(), inplace=True)
                    data_fillna = df.isnull().sum().to_frame(name='Null_values_per_column').to_html()
        return render_template('imputation.html', text_col=text_col, data_fillna=data_fillna)
    except:
        return "Either there are no columns with null values or they have been removed."


@app.route('/show_describe')
def showDes():
    uploaded_df=df.describe()
    uploaded_df_html =uploaded_df.to_html(justify='left')
    return render_template('page1.html', data_des = uploaded_df_html)


@app.route('/show_correlation')
def showCor():
    uploaded_df=df.corr()
    uploaded_df_html =uploaded_df.to_html(justify='left')
    plt.figure(figsize=(10,10))
    sns.heatmap(df.corr(),annot=True)
    plt_file = "static/img3.png"
    plt.savefig(plt_file)
    return render_template('page1.html', data_corr = uploaded_df_html,heatmap =plt_file) 

@app.route('/train')
def train():

    return render_template('train.html')



@app.route('/get_columns', methods=['POST'])
def get_columns():
    columns = df.columns.tolist()
    return render_template('train.html', columns=columns)


@app.route('/choose_target_variables', methods=['POST'])
def choose_variables():
    global target_variable
    global selected_columns
    target_variable = request.form.getlist('options')
    if not target_variable:
        return "Please choose at least one variable."
    selected_df = df.drop(target_variable, axis=1)
    selected_columns = selected_df.columns.tolist()
    print('these are the values ',target_variable, selected_columns)
    return render_template('train.html',columns=df.columns.tolist(), selected_columns=', '.join(selected_columns), target_var=target_variable[0])


@app.route('/split_data' , methods=['GET', 'POST'])
def split():
    global selected_columns
    global target_variable
    global X_train
    global X_test
    global Y_test
    global Y_train

    try:
        df.dropna(inplace=True)
        X = df.loc[:, selected_columns]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Y = df[target_variable]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
    except:
        return "Please choose your target variable"

    X_train=X_train.shape
    X_test=X_test.shape
    Y_train=Y_train.shape
    Y_test=Y_test.shape

    return render_template('train.html',X_train=X_train,X_test =X_test, Y_train=Y_train, Y_test=Y_test)


@app.route('/train_model', methods=['POST'])
def train_model():
    global selected_columns
    global target_variable
    global X_train
    global X_test
    global Y_test
    global y_train_predict
    global Y_train
    global y_test_predict
    algo = request.form.getlist('algorithms')
    if not algo:
        return "Please choose any one option."

    try:
        df.dropna(inplace=True)
        X = df.loc[:, selected_columns]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Y = df[target_variable]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
    except:
        return "Please choose your target variable"

    if algo[0] == 'reg':
        lin_model = LinearRegression()
        lin_model.fit(X_train, Y_train)
        y_train_predict = lin_model.predict(X_train)
        train_rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
        train_r2 = r2_score(Y_train, y_train_predict)
        y_test_predict = lin_model.predict(X_test)
        test_rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
        test_r2 = r2_score(Y_test, y_test_predict)
        plt.figure(figsize=(10,5))
        plt.title("Actual vs Predicted Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.scatter(Y_test, y_test_predict, color = 'blue')
        reg_scatter = 'static/scatter_plot.png'
        plt.savefig(reg_scatter)
        return render_template('train.html', train_r2=train_r2, train_rmse=train_rmse, test_r2=test_r2, test_rmse=test_rmse, reg_scatter=reg_scatter)

    if algo[0] == 'class':
        clf = LogisticRegression()
        clf.fit(X_train, Y_train)
        y_train_predict = clf.predict(X_train)
        train_acc = sm.accuracy_score(Y_train, y_train_predict)
        y_test_predict = clf.predict(X_test)
        test_acc = sm.accuracy_score(Y_test, y_test_predict)
        cm = sm.confusion_matrix(Y_test, y_test_predict)
        cm_df = pd.DataFrame(cm, index=None, columns=None)
        cm_df = cm_df.to_html(index=False, header=False)
        return render_template('train.html', train_acc=train_acc, test_acc=test_acc, cm=cm_df)


@app.route('/plot')
def plot():
    global X_train
    global Y_train
    global X_test
    global Y_test
    global y_train_predict
    global y_test_predict
    # plt.scatter(X_train, Y_train, color = 'red')
    # plt.plot(X_test, y_test_predict, color = 'blue')
    # plt.scatter(Y_train, y_train_predict, color = 'red')
    plt.scatter(X_test, y_test_predict, color = 'blue')
    plt.savefig('plot.png')
    # # return the image file
    return send_file('plot.png', mimetype='image/png')


@app.route('/show_outlier', methods=['GET', 'POST'])
def outlier():
        global df
        df = df
        mean = df.mean()
        median = df.median()
        std = df.std()

        # Calculate the z-scores for each value in the data frame
        z_scores = np.abs((df - df.mean()) / df.std())

        # Filter the data frame to show only the rows with z-scores above a certain threshold
        threshold = 3
        outliers = df[(z_scores > threshold).any(axis=1)]

        # Render the template with the outlier data frame
        return render_template('index2.html',mean=mean,median=median,std=std, count=outliers.shape, outlier_data=outliers.to_html(justify='left'))
  



if __name__ == '__main__':  
    app.run(debug=True)