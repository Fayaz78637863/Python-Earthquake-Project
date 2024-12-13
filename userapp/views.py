from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from userapp.models import *
import urllib.request
import pandas as pd
import time
from adminapp.models import *
import urllib.parse
import random
import ssl


# Create your views here.






# ------------------------------------------------------------------------------------------------



#userviews



import pytz
from datetime import datetime



def user_dashboard(req):
    prediction_count = UserModel.objects.all().count()
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    Feedbacks_users_count = Feedback.objects.all().count()
    all_users_count = UserModel.objects.all().count()

    if user.Last_Login_Time is None:
        IST = pytz.timezone("Asia/Kolkata")
        current_time_ist = datetime.now(IST).time()
        user.Last_Login_Time = current_time_ist
        user.save()
        return redirect("user_dashboard")

    return render(
        req,
        "user/User_dashboard.html",
        {
            "predictions": prediction_count,
            "user_name": user.user_name,  
            "feedback_count": Feedbacks_users_count,
            "all_users_count": all_users_count,
        },
    )


def user_profile(req):
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    if req.method == "POST":
        user_name = req.POST.get("username")
        user_age = req.POST.get("age")
        user_phone = req.POST.get("mobile_number")
        user_email = req.POST.get("email")
        user_password = req.POST.get("Password")
        user_address = req.POST.get("address")

        user.user_name = user_name
        user.user_age = user_age
        user.user_address = user_address
        user.user_contact = user_phone
        user.user_email = user_email
        user.user_password = user_password

        if len(req.FILES) != 0:
            image = req.FILES["profilepic"]
            user.user_image = image
            user.save()
            messages.success(req, "Updated Successfully.")
        else:
            user.save()
            messages.success(req, "Updated Successfully.")

    context = {"i": user}
    return render(req, "user/User_profile.html", context)


# ---------------------------------------------------------------------------------------------------------------


import os
from django.core.files.storage import default_storage
from django.contrib import messages
from django.conf import settings
from django.contrib import messages







def Classification(req):

        return render(req, "user/detection.html")

def Classification_result(req):

            return render(req, "user/detection-result.html",)




# ------------------------------------------------------------------------------


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


from django.contrib import messages
from django.shortcuts import redirect, render
from .models import Feedback, UserModel  # Make sure to import your models


def user_feedback(req):
    id = req.session["user_id"]
    user = UserModel.objects.get(user_id=id)
    
    if req.method == "POST":
        rating = req.POST.get("rating")
        review = req.POST.get("review")
        
        # Sentiment analysis
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)
        
        if score["compound"] > 0 and score["compound"] <= 0.5:
            sentiment = "positive"
        elif score["compound"] >= 0.5:
            sentiment = "very positive"
        elif score["compound"] < -0.5:
            sentiment = "negative"
        elif score["compound"] < 0 and score["compound"] >= -0.5:
            sentiment = "very negative"
        else:
            sentiment = "neutral"
        
        # Create the feedback
        Feedback.objects.create(
            Rating=rating,
            Review=review,
            Sentiment=sentiment,
            Reviewer=user
        )
        
        messages.success(req, "Feedback recorded")
        return redirect("user_feedback")  # Redirecting to the same page
    
    return render(req, "user/user_feedback.html")



from django.utils import timezone


def user_logout(req):
    if "user_id" in req.session:
        view_id = req.session["user_id"]
        try:
            user = UserModel.objects.get(user_id=view_id)
            user.Last_Login_Time = timezone.now().time()
            user.Last_Login_Date = timezone.now().date()
            user.save()
            messages.info(req, "You are logged out.")
        except UserModel.DoesNotExist:
            pass
    req.session.flush()
    return redirect("user_login")



def user_login(req):
    if req.method == "POST":
        user_email = req.POST.get("email")
        user_password = req.POST.get("password")
        print(user_email, user_password)

        try:
            users_data = UserModel.objects.filter(user_email=user_email)
            if not users_data.exists():
                messages.error(req, "User does not exist")
                return redirect("user_login")

            for user_data in users_data:
                if user_data.user_password == user_password:
                    if (
                        user_data.Otp_Status == "verified"
                        and user_data.User_Status == "accepted"
                    ):
                        req.session["user_id"] = user_data.user_id
                        messages.success(req, "You are logged in..")
                        user_data.No_Of_Times_Login += 1
                        user_data.save()
                        return redirect("user_dashboard")
                    elif (
                        user_data.Otp_Status == "verified"
                        and user_data.User_Status == "pending"
                    ):
                        messages.info(req, "Your Status is in pending")
                        return redirect("user_login")
                    else:
                        messages.warning(req, "verifyOTP...!")
                        req.session["user_email"] = user_data.user_email
                        return redirect("otp")
                else:
                    messages.error(req, "Incorrect credentials...!")
                    return redirect("user_login")

            # Handle the case where no user data matched the password
            messages.error(req, "Incorrect credentials...!")
            return redirect("user_login")
        except Exception as e:
            print(e)
            messages.error(req, "An error occurred. Please try again later.")
            return redirect("user_login")

    return render(req, "user/user-login.html")

import pickle
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages



from django.contrib import messages

def Classification(request):
    if request.method == 'POST':
        # Extracting data from the form
        CDI = request.POST.get('CDI')
        NST = request.POST.get('NST')
        Depth = request.POST.get('Depth')
        Magnitude = request.POST.get('Magnitude')
        Latitude = request.POST.get('Latitude')
        Longitude = request.POST.get('Longitude')
        Year = request.POST.get('Year')

        # Check for empty fields and display a warning message
        if not all([CDI, NST, Depth, Magnitude, Latitude, Longitude, Year]):
            messages.warning(request, "Please fill in all input details.")
            return render(request, "user/Prediction.html")

        try:
            # Convert to float
            CDI = float(CDI)
            NST = float(NST)
            Depth = float(Depth)
            Magnitude = float(Magnitude)
            Latitude = float(Latitude)
            Longitude = float(Longitude)
            Year = float(Year)
        except ValueError:
            messages.warning(request, "Please enter valid numbers.")
            return render(request, "user/Prediction.html")

        # Loading the saved RandomForestClassifier model
        file_path = 'xbg_earth.pkl'  # Update with your model file path
        try:
            with open(file_path, 'rb') as file:
                loaded_model = pickle.load(file)
        except FileNotFoundError:
            messages.error(request, "Model file not found.")
            return redirect("Classification")  # Redirect back to the form page

        prediction = loaded_model.predict([[CDI, NST, Depth, Magnitude, Latitude, Longitude, Year]])

        # Convert predictions to standard Python int
        prediction_result = int(prediction[0])

        # Store the prediction result in the session
        request.session['prediction_result'] = prediction_result

        return redirect("Classification_result")  # Replace with your actual redirect URL

    return render(request, "user/Prediction.html")







def Classification_result(request):
    # Retrieve the prediction result from the session
    prediction_result = request.session.get('prediction_result', None)

    # Retrieve the latest XG_ALGO model details from the database
    xg_algo = XG_ALGO.objects.last()

    if xg_algo:
        model_details = {
            'name': xg_algo.Name,
            'accuracy': xg_algo.Accuracy,
            'precision': xg_algo.Precession,
            'f1_score': xg_algo.F1_Score,
            'recall': xg_algo.Recall,
        }
    else:
        model_details = None

    context = {
        'prediction_result': prediction_result,
        'model_details': model_details,
    }

    return render(request, "user/Prediction-result.html", context)


