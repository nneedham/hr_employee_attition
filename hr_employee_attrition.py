# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:20:58 2020

@author: needh
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize']=(30,15)
plt.style.use('fivethirtyeight')

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

#let's get some general information to better understand what we are looking at
df.shape
print(df.columns)

head = df.head()

# We can start removing the columns that don't add anything to our analysis
df = df.drop(['EmployeeCount','EmployeeNumber', 'Over18','StandardHours'], axis = 1)

# Now let's create a df for the numerical variables. Numerical data are measurements or counts
df_num = df[['Age',
             'DailyRate',
             'DistanceFromHome',
             'HourlyRate',
             'MonthlyIncome',
             'MonthlyRate',
             'NumCompaniesWorked',
             'PercentSalaryHike',
             'TotalWorkingYears',
             'TrainingTimesLastYear',
             'YearsAtCompany',
             'YearsInCurrentRole',
             'YearsSinceLastPromotion',
             'YearsWithCurrManager']]

#Let's start by looking at a histogram 
df_num.hist()
## We can look at these histograms for anything that sticks out. 
# Just a few observations:
    # Age follows a normal distribution
    # While Monthly income is what we would expect... there are more people 
    # at lower ranks who earn less... the daily rate, hourly rate and monthly rate 
    # don't match that
    # distance from homw, Monthly Income, Percent Salary Hike, Years at Company, 
    # Years since last promotion seem to follow roughly an expoential distribution

#Next let's look at the boxplots
for i in df_num.columns:
    df_num[[i]].boxplot()
    plt.show()
    
sns.set(font_scale=2)
num_heatmap = sns.heatmap(df_num.corr(), annot=True, cmap='Blues')
num_heatmap.set_xticklabels(num_heatmap.get_yticklabels(), rotation=40)
plt.show()

#Now let's create a df for the categorial variables. This will also include ordinal values such as ratings from 0-5