# Predicting Interview Attendance in India
Given a real world set of data, can I build a model that can predict interview attendance in India. If so, what other data can we extrapolate from the model to inform interested parties.

## Data
This Kaggle Dataset is a real collection of data collected by researchers in order to help employers, recruiters, and other interested parties predict if a candidate is likely to show up for their expected interview. The original data set contains 1,233 rows and 28 columns. The dataset contains various features, such as the client and candidate locations, interview locations, various questions for the interviewee (i.e. can the client follow up with the candidate), the nature of the job, etc.

Because this is a classification problem (predicting a '1' if the candidate will likely show up and a '0' if not), I foresaw a potential data cleaning issue. Each column contained many unique values, and when I created dummy variables without any feature engineering, the dataset exploded into thousands of columns. Therefore, I wanted to be very precise with eliminating features that were not going to help me predict.

The first thing I did was convert the interview date to a date time object so I could extract the months and the day of the weeks. I also combined 7 yes or no questions that candidates answered regarding their preparation and expectations for the interview. I then changed any column that had a true/false or yes/no to 1's and 0's. I also made sure to carefully examine the unique values in every column since this dataset included information that was hand entered by the collector. There were various instances where cities were logged multiple times, but with different spellings or capitalizations. I aggregated this whenever I discovered it.

## EDA
Next, I decided that I wanted to group all the columns and sum the instances of observed attendance divided by expected attendance. This allowed me to visually view different features and see if there were variations across the unique features and their observed attendance percentages.

![Image](/images/screenshot_of_bars.png)

This is an image of the final features that I left in the analysis. However, I iterated through several other features and determined that in many cases there was no discrepancy in attendance. For example, gender and marital status had no differentiation. Additionally, I tried to feature engineer some additional columns which added no differentiation (if the interview location or client location differed from the candidate location). It's clear that the month, day of the week, and number of questions the candidates answered were my most telling features which I extrapolated from other data.

Next, I used the Variance Inflation Factor (VIF) in order to examine collinearity. Even though I was planning on using a random forest classification as my primary model which naturally reduces collinearity, I felt that this was a worthwhile exercise in order to reduce the dimensions and be able to focus on meaningful insights from the data. While the above graphs helped me reduce some features at the start, I was still left with about 60 features after I created dummy variables.

Because I didn't remove one dummy variable at the start from each feature, there was naturally collinearity between all features. The first thing I did was remove one dummy variable in order to address this. However, the majority of my features were still very correlated to each other.  I was able to reduce my dimensions by 32 features using the VIF process.

![image](/images/vif.png)

## Modeling

For this case study, I decided to use precision as my metric which is defined at (True Positives)/(True Positives + False Positives). I thought carefully about each metric and the business implications associated with each one. 

A false positive means that recruiters/employers inaccurately predicted someone to show up for the interview when they ended up being a no show. This would cost the company money, since that time could have been used to do other things such as interview other candidates. I figured if we can reduce the amount of False Positives, that would help improve productivity for companies and allow recruiters to more accurately identify candidates who need reminders to follow up.

The first model I decided to use was the Random Forest Classification because it is generally regarded as a good predictor in classification problems. I used a grid search in order to help identify the optimal hyper parameters.

I compared my results to other classification models such as a Logistic Regression,  Decision Tree, Gradient Boosted Classification, and an AdaBoost Classification.

While feature engineering my data and removing columns, I frequently updated and examined how the precision metric changed over time. Interestingly, it only had minor improvements.

## Results

| Model | Precision | Recall | Accuracy |
| ----------- | ----------- | ------- | -------|
| Random Forest Classification | .753 | .874 | .726 |
| Logistic Regression | .724 | .860 | .699 |
| Decision Tree Classification | .748 | .853 | .713 |
| Gradient Boosted Classification | .750 | .860 | .718 |
| AdaBoosted Classification | .722 | .853 | .685 |

It appears that the Random Forest Classification model yields the best results, but all models predicted very similar results overall.

Also, it appears that my models do a better job at minimizing false negatives (Recall). Recall is defined at (True Positives)/(True Positives + False Negatives). In this case, a false negative identifies someone as unlikely to show up for their interview but then they show up (false positive is the candidate is predicted to show up, but then they don't). I don't think that recall is the best metric to use because I believe that a company would always plan on their candidates showing up unless explicitly told that the candidate was going to cancel. I think it would be unwise for a company to plan their interview schedules based on predictions because if the candidate actually shows up then it would be very embarrassing for the company. 

Overall, my model is better at minimizing false negatives (saying someone won't show up, but they do) than false positives (saying someone will show up, but they don't).

Fortunately, my model's predictions are better than the mean (about 59%) so there is still useful information to gain from this work.

## Take Aways

In addition to constructing a useful classification model for predictions, I hoped to be able to pull useful information from the features as well.  I decided to examine the feature importances, partial dependency plots, and a decision tree.

![Image](/images/screenshot_fi.png)

Interestingly, the number of questions that candidates answered, the month, and the day of the week are the top three important features. All three of these are features that I extrapolated from the original data. Even more interestingly, the number of questions that candidates answered 'yes' in their questionnaire was the most important feature by far.

Here are the partial dependency plots for each of these three features. This helps us understand how each feature impacts observed attendance.

![Image](/images/num_questions_partial_dependency.png)

The more 'yes' answers a candidate indicates on their questionnaire, the more likely they are to show up.

![Image](/images/month_partial_dependency.png)

There is some variability during the time of the year. The partial dependency plot for months indicates that a later month is more likely to indicate a candidate will attend their interview.

![Image](/images/dow_partial_dependency.png)

Also, a candidate is most likely to show up for an interview on a Monday, but least likely to show up on a Wednesday.

Lastly, the benefit of using a decision tree is that it adds interpretability. While the above graphs add more intuitive insight into important features, the decision tree is still interesting to view.

![Image](/images/decisison_tree.png)

# Conclusion

In conclusion, the number of questions that a candidate responds 'yes' to is a very strong indicator on how engaged the candidate is with the interview process. Also, the time of the year and the day of the week are also very important too. Some industries, candidate locations, and interview venues are also important features (but not as important).  My model may prove to be useful for recruiters or employers to follow up and target candidates who may not show up for interviews. This will allow recruiters to potentially increase their commission and spend time focusing efforts on "at risk" candidates, and also allow companies to better utilize their resources and not waste time on no shows.