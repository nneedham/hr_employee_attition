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
# There are several correlations to note:
    # Age, monthly income, total working years, 
    # Years at company, Years in current role, Years since last promotion,
    # and Years with current manager all seem to have some correlation with each other
    # Most of this should be expected since as people age, they will gain experience 
    # which will make them more valuable, but also more specialized so it might 
    # make it more difficult for them to move out of the role they are in
    # num of companies worked for doesn't quite fit into this cluster neatly, 
    # which also should be expected since the longer you work at one company,
    # the less time you have to work at other companies
    # Usually the more companies that you work for the higher the salary, so it 
    # is interesting that there isn't a strong correlation between number of companies worked 
    # for and monthly income. Especially when years in current role and company and
    # since last promotion and with current manager are so low... this indicates
    # that this company strongly rewards loyalty. So a follow on question is
    # whether or not they reward loyatly at the cost of productivity
    # Since there are so many correlations it will be important to factor them
    # into the regression analysis because it can make the coefficients of a 
    # regression model unstable
    # https://blog.exploratory.io/why-multicollinearity-is-bad-and-how-to-detect-it-in-your-regression-models-e40d782e67e
    # So to correct for these multicolinearity we will drop a few columns
    # We will drop all of the aforementioned columns and add each one once when we test regressions

#Now let's create a df for the categorial variables. This will also include ordinal values such as ratings from 0-5
df_cat = df[['Attrition',
             'BusinessTravel',
             'Department',
             'Education',
             'EducationField',
             'EnvironmentSatisfaction',
             'JobInvolvement',
             'JobLevel',
             'JobRole',
             'JobSatisfaction',
             'MaritalStatus',
             'OverTime',
             'PerformanceRating',
             'RelationshipSatisfaction',
             ]]
#check for correlations
sns.set(font_scale=2)
num_heatmap = sns.heatmap(df_cat.corr(), annot=True, cmap='Blues')
num_heatmap.set_xticklabels(num_heatmap.get_yticklabels(), rotation=40)
plt.show()
#Here we have few correlations than before. The strongest being Education and
#Job level which makes sense

#check for barplot
for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_yticklabels(), rotation=45)
    plt.show()
    
#FILL THIS IN LATER... WHAT DO YOU SEE FROM THE BARPLOT?