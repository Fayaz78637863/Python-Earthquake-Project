from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from userapp.models import *
from adminapp.models import *
import urllib.request
import urllib.parse
import pandas as pd
from django.core.paginator import Paginator
from userapp.models import UserModel


# Create your views here.
def admin_dashboard(req):
    all_users_count = UserModel.objects.all().count()
    pending_users_count = UserModel.objects.filter(User_Status="pending").count()
    rejected_users_count = UserModel.objects.filter(User_Status="removed").count()
    accepted_users_count = UserModel.objects.filter(User_Status="accepted").count()
    Feedbacks_users_count = Feedback.objects.all().count()
    prediction_count = UserModel.objects.all().count()
    return render(
        req,
        "admin/admin-dashboard.html",
        {
            "a": all_users_count,
            "b": pending_users_count,
            "c": rejected_users_count,
            "d": accepted_users_count,
            "e": Feedbacks_users_count,
            "f": prediction_count,
        },
    )


def pending_users(req):
    pending = UserModel.objects.filter(User_Status="pending")
    paginator = Paginator(pending, 5)
    page_number = req.GET.get("page")
    post = paginator.get_page(page_number)
    return render(req, "admin/Pending-users.html", {"user": post})


def all_users(req):
    all_users = UserModel.objects.all()
    paginator = Paginator(all_users, 5)
    page_number = req.GET.get("page")
    post = paginator.get_page(page_number)
    return render(req, "admin/All-Users.html", {"allu": all_users, "user": post})


def delete_user(request, user_id):
    try:
        user = UserModel.objects.get(user_id=user_id)
        user.delete()
        messages.warning(request, "User was deleted successfully!")
    except UserModel.DoesNotExist:
        messages.error(request, "User does not exist.")
    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")
    
    return redirect("all_users")


# Acept users button
def accept_user(request, id):
    try:
        status_update = UserModel.objects.get(user_id=id)
        status_update.User_Status = "accepted"
        status_update.save()
        messages.success(request, "User was accepted successfully!")
    except UserModel.DoesNotExist:
        messages.error(request, "User does not exist.")
    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")
    
    return redirect("pending_users")




# Remove user button
def reject_user(req, id):
    status_update2 = UserModel.objects.get(user_id=id)
    status_update2.User_Status = "removed"
    status_update2.save()
    messages.warning(req, "User was Rejected..!")
    return redirect("pending_users")

# Change status users button
def change_status(request, id):
    user_data = UserModel.objects.get(user_id=id)
    if user_data.User_Status == "removed":
        user_data.User_Status = "accepted"
        user_data.save()
    elif user_data.User_Status == "accepted":
        user_data.User_Status = "removed"
        user_data.save()
    elif user_data.User_Status == "pending":
        messages.info(request, "Accept the user first..!")
        return redirect ("all_users")
    messages.success(request, "User status was changed..!")
    return redirect("all_users")




def adminlogout(req):
    messages.info(req, "You are logged out.")
    return redirect("admin_login")




def upload_dataset(request):
    if request.method == 'POST':
        file = request.FILES['data_file']
        file_size = str((file.size)/1024) +' kb'
        Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(request, 'Your dataset was uploaded..')
    return render(request, "admin/Upload-dataset.html")

# Admin delete dataset button
def delete_dataset(request, id):
    dataset = Upload_dataset_model.objects.get(user_id = id).delete()
    messages.warning(request, 'Dataset was deleted..!')
    return redirect('view_dataset')

def view_dataset(request):
    dataset = Upload_dataset_model.objects.all()
    paginator = Paginator(dataset, 5)
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request, "admin/View_dataset.html", {'data' : dataset, 'user' : post})

def view_view_dataset(request):
    # df=pd.read_csv('heart.csv')
    data = Upload_dataset_model.objects.last()
    print(data,type(data),'sssss')
    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    table = df.to_html(table_id='data_table')
    return render(request,'admin/View_view_dataset.html', {'t':table})



def ADABoost(request):
    return render(request, "admin/Ada-boost.html")


# 
import pandas as pd
from django.shortcuts import render
from django.contrib import messages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from .models import Upload_dataset_model, ADA_ALGO

def ADABoost_btn(req):
    from django.shortcuts import render
from django.contrib import messages
from .models import Upload_dataset_model
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from django.shortcuts import render
from django.contrib import messages
from .models import Upload_dataset_model, ADA_ALGO  # Ensure ADA_ALGO is imported
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd

def ADABoost_btn(request):
    # Retrieve the latest uploaded dataset
    data = Upload_dataset_model.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        return render(request, "admin/Ada-boost-part2.html")

    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    
    # Assume that the last column is the target variable
    X = df.drop('tsunami', axis=1)
    y = df['tsunami']
    
    # Handle class imbalance
    rs = RandomOverSampler(random_state=42)
    X, y = rs.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the AdaBoost Classifier
    ada = AdaBoostClassifier(random_state=42)
    ada.fit(X_train, y_train)

    # Make predictions
    train_prediction = ada.predict(X_train)
    test_prediction = ada.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_prediction)*100
    test_accuracy = accuracy_score(y_test, test_prediction)*100
    precision = precision_score(y_test, test_prediction, average='weighted')
    recall = recall_score(y_test, test_prediction, average='weighted')
    f1 = f1_score(y_test, test_prediction, average='weighted')

    # Save results to the database
    name = "Ada boost "
    ADA_ALGO.objects.create(
        Accuracy=test_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = ADA_ALGO.objects.last()
    messages.success(request, 'Algorithm executed successfully')

    return render(request, 'admin/Ada-boost-part2.html', {
        'i': latest_algo,
    })

def Random_Forest(request):
    return render(request, "admin/Random-forest.html")
from sklearn.ensemble import RandomForestClassifier
from django.shortcuts import render
from django.contrib import messages
from .models import Upload_dataset_model, RANDOM_ALGO  # Ensure RANDOM_ALGO is imported
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def RandomForest_btn(request):
    # Retrieve the latest uploaded dataset
    data = Upload_dataset_model.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        return render(request, "admin/Random-forest-part2.html")

    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    
    # Assume that the last column is the target variable
    X = df.drop('tsunami', axis=1)
    y = df['tsunami']
    
    # Handle class imbalance
    rs = RandomOverSampler(random_state=42)
    X, y = rs.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForest Classifier
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)

    # Make predictions
    train_prediction = rfc.predict(X_train)
    test_prediction = rfc.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_prediction)*100
    test_accuracy = accuracy_score(y_test, test_prediction)*100
    precision = precision_score(y_test, test_prediction, average='weighted')
    recall = recall_score(y_test, test_prediction, average='weighted')
    f1 = f1_score(y_test, test_prediction, average='weighted')
    class_report = classification_report(y_test, test_prediction)

    # Save results to the database
    name = "Ada boost "
    RANDOM_ALGO.objects.create(
        Accuracy=test_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = RANDOM_ALGO.objects.last()
    messages.success(request, 'Algorithm executed successfully')

    return render(request, 'admin/Random-forest-part2.html', {
        'i': latest_algo,
    })


def Decision_tree(request):
    return render(request, "admin/Decision-tree.html")



from sklearn.tree import DecisionTreeClassifier

def Decision_tree_btn(request):
    # Retrieve the latest uploaded dataset
    data = Upload_dataset_model.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        return render(request, "admin/Decision-tree-part2.html")

    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    
    # Assume that the last column is the target variable
    X = df.drop('tsunami', axis=1)
    y = df['tsunami']
    
    # Handle class imbalance
    rs = RandomOverSampler(random_state=42)
    X, y = rs.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree Classifier
    DT = DecisionTreeClassifier(random_state=42)
    DT.fit(X_train, y_train)

    # Make predictions
    train_pred = DT.predict(X_train)
    test_pred = DT.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_pred)*100
    test_accuracy = accuracy_score(y_test, test_pred)*100
    precision = precision_score(y_test, test_pred, average='weighted')
    recall = recall_score(y_test, test_pred, average='weighted')
    f1 = f1_score(y_test, test_pred, average='weighted')
    class_report = classification_report(y_test, test_pred)

        # Save results to the database
    name = "Ada boost "
    DECISION_ALGO.objects.create(
        Accuracy=test_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = DECISION_ALGO.objects.last()
    messages.success(request, 'Algorithm executed successfully')

    return render(request, 'admin/Decision-tree-part2.html', {
        'i': latest_algo,
    })

def Gradient_boosting(request):
    return render(request, "admin/Gradient-decent.html")

from django.shortcuts import render
from django.contrib import messages
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from .models import GRADIENT_ALGO, Upload_dataset_model

def Gradient_boosting_btn(request):
    # Retrieve the latest uploaded dataset
    data = Upload_dataset_model.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        return render(request, "admin/Gradient-decent-part2.html")

    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    
    # Assume that the last column is the target variable
    X = df.drop('tsunami', axis=1)
    y = df['tsunami']
    rs = RandomOverSampler(random_state=42)
    X, y = rs.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Gradient Boosting Classifier
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)

    # Make predictions
    train_prediction = gb.predict(X_train)
    test_prediction = gb.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_prediction)*100
    test_accuracy = accuracy_score(y_test, test_prediction)*100
    precision = precision_score(y_test, test_prediction, average='weighted')
    recall = recall_score(y_test, test_prediction, average='weighted')
    f1 = f1_score(y_test, test_prediction, average='weighted')
    class_report = classification_report(y_test, test_prediction)

    # Save results to the database
    name = "GRADIENT Boost Algorithm"
    GRADIENT_ALGO.objects.create(
        Accuracy=test_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = GRADIENT_ALGO.objects.last()
    messages.success(request, 'Algorithm executed successfully')

    return render(request, 'admin/Gradient-decent-part2.html', {
        'i': latest_algo,
    })




def XG_Boost(request):
    
    return render(request, "admin/XG-boost.html")

from django.shortcuts import render
from django.contrib import messages
from .models import Upload_dataset_model, XG_ALGO  # Ensure XG_ALGO is imported
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import pandas as pd

def XG_Boost_btn(request):
    data = Upload_dataset_model.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        return render(request, "admin/XG-boost-part2.html")

    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    
    # Assume that the last column is the target variable
    X = df.drop('tsunami', axis=1)
    y = df['tsunami']
    rs = RandomOverSampler(random_state=42)
    X, y = rs.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost Classifier
    xg = XGBClassifier(random_state=42)
    xg.fit(X_train, y_train)

    # Make predictions
    train_prediction = xg.predict(X_train)
    test_prediction = xg.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_prediction)*100
    test_accuracy = accuracy_score(y_test, test_prediction)*100
    precision = precision_score(y_test, test_prediction, average='weighted')
    recall = recall_score(y_test, test_prediction, average='weighted')
    f1 = f1_score(y_test, test_prediction, average='weighted')
    class_report = classification_report(y_test, test_prediction)

    # Save results to the database
    name = "XG Boost Algorithm"
    XG_ALGO.objects.create(
        Accuracy=test_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest XG_ALGO entry
    latest_algo = XG_ALGO.objects.last()
    messages.success(request, 'Algorithm executed successfully')

    return render(request, 'admin/XG-boost-part2.html', {
        'i': latest_algo,
    })




def Logistic_Regression(request):
    return render(request, "admin/logistic-regression.html")

from django.shortcuts import render
from django.contrib import messages
from .models import Upload_dataset_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

def Logistic_Regression_btn(request):
    # Retrieve the latest uploaded dataset
    data = Upload_dataset_model.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        return render(request, "admin/Logistic-regression-part2.html")

    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    
    # Assume that the last column is the target variable
    X = df.drop('tsunami', axis=1)
    y = df['tsunami']
    rs = RandomOverSampler(random_state=42)
    X, y = rs.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    # Make predictions
    train_prediction = lr.predict(X_train)
    test_prediction = lr.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_prediction)*100
    test_accuracy = accuracy_score(y_test, test_prediction)*100
    precision = precision_score(y_test, test_prediction, average='weighted')
    recall = recall_score(y_test, test_prediction, average='weighted')
    f1 = f1_score(y_test, test_prediction, average='weighted')
    class_report = classification_report(y_test, test_prediction)

    # Prepare results for rendering
    # Save results to the database
    name = "GRADIENT Boost Algorithm"
    LOGISTIC_ALGO.objects.create(
        Accuracy=test_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = LOGISTIC_ALGO.objects.last()
    messages.success(request, 'Algorithm executed successfully')

    return render(request, 'admin/Logistic-regression-part2.html', {
        'i': latest_algo,
    })

    


def Knn(request):
    return render(request, "admin/KNN.html")

from django.shortcuts import render
from django.contrib import messages
from .models import Upload_dataset_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

def Knn_btn(request):
    # Retrieve the latest uploaded dataset
    data = Upload_dataset_model.objects.last()
    if data is None:
        messages.error(request, 'No dataset available.')
        

    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    
    # Assume that the last column is the target variable
    X = df.drop('tsunami', axis=1)
    y = df['tsunami']
    rs = RandomOverSampler(random_state=42)
    X, y = rs.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the K-Nearest Neighbors model
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    # Make predictions
    train_prediction = knn.predict(X_train)
    test_prediction = knn.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_prediction)*100
    test_accuracy = accuracy_score(y_test, test_prediction)*100
    precision = precision_score(y_test, test_prediction, average='weighted')
    recall = recall_score(y_test, test_prediction, average='weighted')
    f1 = f1_score(y_test, test_prediction, average='weighted')
    class_report = classification_report(y_test, test_prediction)

    # Prepare results for rendering
    # Save results to the database
    name = "GRADIENT Boost Algorithm"
    KNN_ALGO.objects.create(
        Accuracy=test_accuracy,
        Precession=precision,
        F1_Score=f1,
        Recall=recall,
        Name=name
    )

    # Retrieve the latest GRADIENT_ALGO entry
    latest_algo = KNN_ALGO.objects.last()
    messages.success(request, 'Algorithm executed successfully')

    return render(request, 'admin/KNN-part2.html', {
        'i': latest_algo,
    })


import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd
from django.shortcuts import render
from django.contrib import messages
from .models import Upload_dataset_model  

def create_histogram(df, ax):
    if 'latitude' in df.columns:
        ax.hist(df['latitude'].dropna(), bins=20, color='skyblue', edgecolor='black')
        ax.set_title('Histogram of Latitude')
    else:
        ax.set_title('Column "latitude" not found')

def create_boxplot(df, ax):
    if 'longitude' in df.columns:
        sns.boxplot(df['longitude'].dropna(), color='green', ax=ax)
        ax.set_title('Boxplot of Longitude')
        ax.set_xlabel('longitude')
    else:
        ax.set_title('Column "longitude" not found')

def create_lineplot(df, ax):
    if 'latitude' in df.columns and 'longitude' in df.columns:
        sns.lineplot(x='latitude', y='longitude', data=df, ax=ax)
        ax.set_title('Line Plot of Longitude vs. Latitude')
        ax.set_xlabel('latitude')
        ax.set_ylabel('longitude')
    else:
        ax.set_title('Required columns "latitude" or "longitude" not found')

def create_piechart(df, ax):
    if 'tsunami' in df.columns:
        ax.pie(df["tsunami"].value_counts().values, labels=df["tsunami"].value_counts().index, autopct="%.02f%%")
        ax.set_title("Tsunami Prediction")
    else:
        ax.set_title('Column "tsunami" not found')

def plot_to_base64(fig):
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig) 
    return f'data:image/png;base64,{img_data}'

def Exploration(request):
    # Check if there is any dataset uploaded
    if not Upload_dataset_model.objects.exists():
        messages.error(request, 'Upload Dataset First.')
        return render(request, 'admin/Upload-dataset.html', {})
    
    # Retrieve the latest uploaded dataset
    dataset = Upload_dataset_model.objects.last()
    try:
        df = pd.read_csv(dataset.Dataset.path)
    except Exception as e:
        messages.error(request, f'Error reading dataset: {e}')
        return render(request, 'admin/Upload-dataset.html', {})

    # Create subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # Plot the Histogram
    create_histogram(df, axes[0, 0])

    # Plot the Boxplot
    create_boxplot(df, axes[0, 1])

    # Plot the Line Plot
    create_lineplot(df, axes[1, 0])

    # Plot the Pie Chart
    create_piechart(df, axes[1, 1])

    # Convert the entire figure to base64-encoded image for rendering in HTML
    figure_img = plot_to_base64(fig)

    # Close the figure to free up resources
    plt.close(fig)

    messages.success(request, 'Data Exploration Analysis Completed Successfully.')
    return render(request, 'admin/Exploration.html', {
        'figure_img': figure_img,
        'dataset': df.to_html(),  # Add the dataset to the context
    })



from django.shortcuts import render
from .models import XG_ALGO, ADA_ALGO, RANDOM_ALGO, GRADIENT_ALGO,KNN_ALGO,DECISION_ALGO,LOGISTIC_ALGO

from django.contrib import messages

def Comparision(request):
    # Retrieve accuracy values from the database for all algorithms
    xg_details = XG_ALGO.objects.last()
    ada_details = ADA_ALGO.objects.last()
    random_details = RANDOM_ALGO.objects.last()
    Knn_details = KNN_ALGO.objects.last()
    Logistic_Regression = LOGISTIC_ALGO.objects.last()
    gradient_details = GRADIENT_ALGO.objects.last()
    decision_details = DECISION_ALGO.objects.last()  

    # Check if any model details are None
    if not all([xg_details, ada_details, random_details, Logistic_Regression, Knn_details, gradient_details, decision_details]):
        messages.error(request, 'Run the Algorithms First.')
        return redirect('Random_Forest')  

    context = {
        'xg_accuracy': float(xg_details.Accuracy) if xg_details else 0,
        'ada_accuracy': float(ada_details.Accuracy) if ada_details else 0,
        'random_accuracy': float(random_details.Accuracy) if random_details else 0,
        'logistic_accuracy': float(Logistic_Regression.Accuracy) if Logistic_Regression else 0,
        'knn_accuracy': float(Knn_details.Accuracy) if Knn_details else 0,
        'gradient_accuracy': float(gradient_details.Accuracy) if gradient_details else 0,
        'decision_accuracy': float(decision_details.Accuracy) if decision_details else 0,
    }

    return render(request, 'admin/Comparison.html', context)

    


def admin_Feedback(request):
    feed = Feedback.objects.all()
    return render(request, "admin/admin-Feedback.html", {"back": feed})
    

def admin_Sentimet_analysis(request):
    fee = Feedback.objects.all()
    return render(request, "admin/Sentiment-analysis.html", {"cat": fee})
    

def admin_Sentimet_analysis_graph(request):
    positive = Feedback.objects.filter(Sentiment="positive").count()
    very_positive = Feedback.objects.filter(Sentiment="very positive").count()
    negative = Feedback.objects.filter(Sentiment="negative").count()
    very_negative = Feedback.objects.filter(Sentiment="very negative").count()
    neutral = Feedback.objects.filter(Sentiment="neutral").count()
    context = {
        "vp": very_positive,
        "p": positive,
        "neg": negative,
        "vn": very_negative,
        "ne": neutral,
    }
    return render(request, "admin/Sentiment-analysis-graph.html",context)



