#!/usr/bin/env python
# coding: utf-8

# importing libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().system('pip install dataprep')
from dataprep.eda import plot


# loading dataset

# In[2]:


data = pd.read_csv('DataCoSupplyChainDataset.csv',header= 0,encoding='unicode_escape')
pd.set_option('display.max_columns',None) #to display all the columns in the dataset
data.head()


# In[55]:


data.info()  #reading the particularities of the dataset


# Selecting Features on the basis of relevance to target

# In[26]:


#the feature were selected through recursive feature elimination 

new_dataset_features = ['Days for shipment (scheduled)','Order Status','Benefit per order','Sales per customer','Category Name',
                        'Market','Customer Segment','Order Item Quantity','Order Region','Customer Country',
                        'Product Price','Shipping Mode', 'Late_delivery_risk']
df = data[new_dataset_features]


# Univariate Analysis
# - 

# Benefit per order

# In[4]:


# Checking box-plot of benefit per order

sns.boxplot(data = df, y = 'Benefit per order', palette = 'rocket')
plt.title('Boxplot - Benefit per order (with outliers)');
plt.xlabel('');


# Checking IQR in Benefir per Order

# In[5]:


Q1 = df['Benefit per order'].quantile(.25)
Q3 = df['Benefit per order'].quantile(.75)
IQR = Q3 - Q1
UL = Q3 + 1.5*(IQR)
LL = Q3 - 1.5*(IQR)
print(UL)
print(LL)


# In[52]:


# We will use numpy to remove outliers

df['Benefit per order'] = np.where(df['Benefit per order'] < -21.89, -21, np.where(df['Benefit per order'] > 151.5, 151, df['Benefit per order']))


# In[40]:


# Check boxplot of Benefit per order after removing outliers

sns.boxplot(data = df, y = 'Benefit per order', palette = 'rocket')
plt.title('Boxplot - Benefit per order (without outliers)');
plt.xlabel('');

# Now we cannot see outliers in Benefit per order


# In[41]:


# Benefit per order (Distribution / Hisplot)

plt.figure(figsize = (10,5))
sns.distplot(df['Benefit per order'], color = 'teal')
plt.title('Distribution of Benefit per order');
plt.xlabel('');
plt.ylabel('');


# Sales per customer 

# In[9]:


# Check outliers in Sales per customer

sns.boxplot(data = df, y = 'Sales per customer', palette = 'rocket')
plt.title('Boxplot - Sales per customer (with outliers)');
plt.xlabel('');


# In[10]:


#from the box plot we can see the existence of many outliers

# Finding outliers in Sales per customer from the dataset

Q1 = df['Sales per customer'].quantile(.25)
Q3 = df['Sales per customer'].quantile(.75)
IQR = Q3 - Q1
UL = Q3 + 1.5*(IQR)
LL = Q3 - 1.5*(IQR)
print(UL)
print(LL)


# In[46]:


# Removing outliers using numpy

df['Sales per customer'] = np.where(df['Sales per customer'] < 32.869, 32, np.where(df['Sales per customer'] > 461.930, 461, df['Sales per customer']))


# In[47]:


# Checking boxplot after removing outliers

sns.boxplot(data = df, y = 'Sales per customer', palette = 'rocket')
plt.title('Boxplot - Sales per customer (without outliers)');
plt.xlabel('');


# In[13]:


# Sales per customer Violin-plot (a hybrid of a box plot and a kernel density plot, which shows peaks in the data)

sns.catplot(data = df, y = 'Sales per customer', kind = 'violin', color = 'yellow')
plt.title('Sales per customer - Violin');
plt.ylabel('');


# In[ ]:


#from teh violin plot we see that the sales per customer peakes around the 100 mark. 
#Sales amounting >200 saw a decreasing trend


# Delivery Status

# In[15]:


#plotting a pie to check the potions of delivery status

pie = data['Delivery Status'].value_counts().plot(kind = 'pie', autopct = '%.2f%%', explode = [0.1, 0.1, 0.1, 0.2], cmap = "Set2")
plt.title('Delivery Status')
plt.ylabel('');


# In[ ]:


#we see that majority of the deliveries in the supply chain have been late  
#the next biggest portion belonged to deliveries that were ordered while paying advanced shipping


# Category Name (Countplot)

# In[16]:


#plotting count plot

plt.figure(figsize = (10,5))
a = df['Category Name'].value_counts(ascending = False)
order = list(a.index[:9]) + list(a.index[-14:])
sns.countplot(data = df, x = 'Category Name', order = order, palette = "crest")
plt.xticks(rotation = 90);
plt.title('Categories by Count')
plt.ylabel('Count');
plt.xlabel('Category');

# To remove 'Others' from plot, we use indexing in orders


# Customer Country

# In[17]:


#plotting count plot for customer country

plt.title('Customer Country - Puerto Rico vs United States')
sns.countplot(data = df, y = 'Customer Country', palette = "crest")
plt.ylabel('Customer Country');

#we see that majorty of the customers are from the US


# Customer Segment (Pie)

# In[18]:


#plotting a pit plot of customer segments

plt.title('Share of Customer Segment')
df['Customer Segment'].value_counts().plot(kind = 'pie', autopct = '%.2f%%', cmap = 'Set2', explode = [0.1, 0.1, 0.1])
plt.ylabel('');

#the largest segment is the consumer segment


# Market (Pie)

# In[19]:


#pie plot to check market share

df['Market'].value_counts().plot(kind = 'pie', autopct = '%.2f%%', cmap = 'Set2', explode = [0.1, 0.1, 0.1, 0.1, 0.1])
plt.title('Share of Market')
plt.ylabel('');


# Sales

# In[23]:


sns.boxplot(data = df, y = 'Sales per customer', palette = 'rocket')
plt.title('Boxplot - Sales (with outliers)');
plt.xlabel('');
plt.ylabel('');


# Order Region

# In[24]:



plt.figure(figsize = (10,5))
a = df['Order Region'].value_counts(ascending = False)
sns.countplot(data = df, x = 'Order Region', palette = 'magma', order = a.index)
plt.title('Order Region');
plt.xlabel('Order Region');
plt.ylabel('Count');
plt.xticks(rotation = 90);


# Order Status

# In[27]:


color = sns.light_palette("seagreen", reverse=True)
plt.figure(figsize = (10,5))
a = df['Order Status'].value_counts(ascending = False)
sns.countplot(data = df, x = 'Order Status', order = a.index, palette = color)
plt.xticks(rotation = 45);
plt.xlabel('Order Status');
plt.ylabel('Count');
plt.title('Orders by Status');


# Shipping Mode

# In[28]:


df['Shipping Mode'].value_counts().plot(kind = 'pie', autopct = '%.2f%%', cmap = 'Set2', explode = [0.1, 0.1, 0.1, 0.1])
plt.title('Share of Shipping Mode')
plt.xlabel('');
plt.ylabel('');


# Bivariate Analysis
# - 

# In[35]:


df['Benefit per order'] = df['Benefit per order'].round(2)


# Benefit per order vs Category Name and department name

# In[42]:


fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

# Subplot 1 - Benefit per order vs Category Name (Boxplot)
sns.boxplot(data = df, x = 'Category Name', y = 'Benefit per order', palette = 'Set2', ax = axes[0]);
axes[0].set_title('Benefit Per Order of Categories');
axes[0].set_xlabel('Category Name');
axes[0].set_ylabel('Benefit Per Order');
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90);

# Subplot 2 - Benefit per order vs Department Name (Violinplot)
sns.violinplot(data = data, y = 'Benefit per order', x = 'Department Name', ax = axes[1], palette = 'Set2', aspect = 1.5)
axes[1].set_xlabel('Department Name', fontsize = 12);
axes[1].set_ylabel('Benefit Per Order', fontsize = 11);
axes[1].set_title('Benefit Per Order of Departments', fontsize = 12);
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90);

# Catplot produces a facetgrid. Here plt.figure(figsize()) doesn't work. Hence we increased label size using fontsize).
# Note: Fontsize doesn't exist in sns.catplot(). Hence we add them on plt.xlabel(), plt.ylabel(), and plt.title().


# Benefit per order vs Market and Benefit per order by Customer Segment (distribution)

# In[43]:




fig, axes = plt.subplots(ncols = 2, figsize = (16,6))

# Subplot 1 - Benefit per order vs Market (Barplot)
sns.barplot(data = df, y = 'Market', x = 'Benefit per order', palette = 'Dark2', ax = axes[0])
axes[0].set_title('Benefit Per Order by Market');
axes[0].set_ylabel('Market');
axes[0].set_xlabel('Benefit Per Order');

# Subplot 2 - Distribution of Benefit per order with Customer Segment (Histplot)
sns.histplot(data = df, x = 'Benefit per order', hue = 'Customer Segment', ax = axes[1])
axes[1].set_title('Benefit per order by Customer Segment');
axes[1].set_xlabel('Benefit Per Order');
axes[1].set_ylabel('');


# Sales per customer vs Caregory and Department

# In[49]:


fig, axes = plt.subplots(ncols = 2, figsize = (16,6))

# Subplot 1 - Sales per customer vs Category Name (Barplot)
a = df.groupby(['Category Name'])['Sales per customer'].mean().sort_values(ascending = False)
sns.barplot(data = df, x = a.index, y = a.values, ci = None, palette = 'Spectral', ax = axes[0])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
axes[0].set_xlabel('Category Name');
axes[0].set_ylabel('Sales per customer');
axes[0].set_title(' Mean Sales Per Customer by Categories');

# Subplot 2 - Sales per customer vs Department Name (Stripplot)
sns.stripplot(x = 'Department Name', y = 'Sales per customer', data = data, ax = axes[1]);
axes[1].set_title('Departments by Sales Per Customer');
axes[1].set_xlabel('Department Name');
axes[1].set_ylabel('Sales per customer');
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation = 90);


# Sales per customer vs Market and distribition with Customer Segment

# In[48]:


fig, axes = plt.subplots(ncols = 2, figsize = (16,6))

# Subplot 1 - Sales per customer vs Market (Barplot)
sns.barplot(data = df, y = 'Market', x = 'Sales per customer', palette = 'viridis', ax = axes[0]);
axes[0].set_title('Sales Per Customer by Markets');
axes[0].set_xlabel('Sales Per Customer');
axes[0].set_ylabel('Markets');

# Subplot 2 - Distribition of Sales per customer with Customer Segment (Histplot)
sns.histplot(data = df, x = 'Sales per customer', hue = 'Customer Segment', ax = axes[1])
axes[1].set_title('Distribution of Sales per customer with Customer Segment');
axes[1].set_xlabel('Sales Per Customer');
axes[1].set_ylabel('');


# Multivariate Analysis
# - 

# Benefit per order vs Market with Type

# In[53]:


sns.catplot(data = data, x = 'Market', y = 'Benefit per order', hue = 'Type', height = 4, aspect = 3, palette = 'colorblind')
plt.title('Markets by Benefit Per Order with Type of Transaction');
plt.xlabel('Market');
plt.ylabel('Benefit Per Order');


# Order Item Profit Ratio v Department & Delivery and Benefit per order vs Shipping Mode & Segment

# In[57]:


fig, axes = plt.subplots(ncols = 2, figsize = (16,6))

# Subplot 1 - Order Item Profit Ratio vs Department and Delivery Status (Barplot) 
sns.barplot(data=data, y='Department Name', x='Order Item Profit Ratio', hue='Delivery Status', ci = None, palette = 'colorblind', ax = axes[0])
axes[0].set_title('Order Item Profit Ratio by Department and Delivery Status');
axes[0].set_xlabel('Order Item Profit Ratio');
axes[0].set_ylabel('Department Name');

# Subplot 2 - Benefit per order vs Shipping Mode and Customer Segment
sns.barplot(data = data, x = 'Shipping Mode', y = 'Benefit per order', hue = 'Customer Segment', palette = 'husl', ax = axes[1], ci = None)
axes[1].set_title('Benefit Per Order by Shipping Mode and Customer Segment');
axes[1].set_xlabel('Shipping Mode');
axes[1].set_ylabel('Benefit Per Order');


# Sales vs Order by Region with Shipping Mode

# In[59]:


plt.figure(figsize = (12,5))
sns.pointplot(data = data, x ='Order Region', y='Sales', hue='Shipping Mode')
plt.xticks(rotation = 90);
plt.title('Shipping Mode of Order Regions by Sales');
plt.xlabel('Order Region');
plt.ylabel('Sales');


# Encoding Varibles
# - 

# In[7]:


prediction_data=pd.get_dummies(data[['Order Status','Shipping Mode', 'Market', 'Customer Segment']], drop_first=False)

#prediction_data=pd.get_dummies(data['Order Status'], drop_first=True)


# In[8]:


df=df.drop(columns=['Order Status','Shipping Mode', 'Market', 'Customer Segment'])


# In[9]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encode the categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        prediction_data[col] = le.fit_transform(df[col])
    else:
        prediction_data[col] = df[col]


# In[10]:


prediction_data.info()


# checking for null values

# In[11]:


prediction_data.isnull().sum()


# In[12]:


prediction_data=prediction_data.dropna(axis=0)


# Implementing ML Algorithms
# - 

# logarithmic regression

# In[13]:


#importind libraries

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#splitting dataset into train and test datasets keeping 'late delivery risk' as the target variable

X_train, X_test, Y_train, Y_test=train_test_split(prediction_data.drop('Late_delivery_risk',axis=1),prediction_data['Late_delivery_risk'],test_size=0.3)

#fitting the logistic regression algorthim on the train dataset
LogReg=LogisticRegression()
LogReg.fit(X_train, Y_train)

#Checking the Prediction accuracy
LogReg.score(X_test, Y_test)

#making predictions using the fitted model
Y_pred=LogReg.predict(X_test)

#displaying classification report as the target variable is categorical in nature
print (classification_report(Y_test,Y_pred))


# Random Forest Classifier

# In[14]:


#loading random forest classifier

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

#fitting the classifier on the dataset
classifier.fit(X_train,Y_train)

#checking score of fitting
classifier.score(X_test,Y_test)

#making predictions
Y_pred_RF=classifier.predict(X_test)

#displaying classification report
print (classification_report(Y_test,Y_pred_RF))


# Gradient Boosting Classifier

# In[15]:


from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

#setting model hyperparameters

gbc_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    subsample=0.9,
    random_state=42
)

#fitting the model
gbc_model.fit(X_train, Y_train)

#making predictions
y_train_pred_gbc = gbc_model.predict(X_train)
y_test_pred_gbc = gbc_model.predict(X_test)

# Calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(Y_train, y_train_pred_gbc)
test_acc = accuracy_score(Y_test, y_test_pred_gbc)

# Print the accuracy scores
print('Train accuracy: ', train_acc)
print('Test accuracy: ', test_acc)

from sklearn.metrics import classification_report

print(classification_report(Y_test,y_test_pred_gbc))


# Monte Carlo Simulation Modelling
# - 

# Checking KDE plots of each variable

# In[17]:


#as the simulations need to run on data that follows real world scenarios,we need to generate 
#random numbers following the probability distributions of the original feature in the dataset

#conducting AutoEDA to check the Kernal Density Estimations of each feature

columns=[col for col in prediction_data.columns ]
for column in columns:
    print(f'\t\t\t\t {column}')
    plot(prediction_data,column).show()
    print('------------------------------------------------------------------------------------------------------------------')


# In[18]:


#looking at data again 

prediction_data.head(2)


# In[19]:



prediction_data.info()


# Checking on the basis of market
# - 

# Scenario 1
# Market= Africa

# In[20]:


#importing libraries

import random
import numpy as np

#writing function to generate values accoring the real life probability distributions and scenario requirements

def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
  
        ret.append([
                 random.randint(0, 1),  #Order Status_CANCELED
                 random.randint(0, 1),  #Order Status_CLOSED   
                 random.randint(0, 1),  #Order Status_COMPLETE
                 random.randint(0, 1),  #Order Status_ON_HOLD 
                 random.randint(0, 1),  #Order Status_PAYMENT_REVIEW
                 random.randint(0, 1),  #Order Status_PENDING    
                 random.randint(0, 1),  #Order Status_PENDING_PAYMENT
                 random.randint(0, 1),  #Order Status_PROCESSING 
                 random.randint(0, 1),  #Order Status_SUSPECTED_FRAUD
                 random.randint(0, 1),  #Shipping Mode_First Class
                 random.randint(0, 1),  #Shipping Mode_Same Day 
                 random.randint(0, 1),  #Shipping Mode_Second Class
                 random.randint(0, 1),  #Shipping Mode_Standard Class
                 1,                     #Market_Africa
                 random.randint(0, 1),  #Market_Europe
                 random.randint(0, 1),  #Market_LATAM
                 random.randint(0, 1),  #Market_Pacific Asia
                 random.randint(0, 1),  #Market_USCA
                 random.randint(0, 1),  #Customer Segment_Consumer
                 random.randint(0, 1),  #Customer Segment_Corporate
                 random.randint(0, 1),  #Customer Segment_Home Office
                 random.randint(0, 4),  #Days for shipment (scheduled)
                 random.uniform(-500.0, 500.0), #Benefit per order
                 random.uniform(5.0, 2000.0),   #Sales per customer
                 random.randint(0, 49), #Category Name
                 random.randint(1, 5),  #Order Item Quantity
                 random.randint(0, 22), #Order Region
                 random.randint(0, 1),  #Customer Country
                 random.uniform(8.0, 2000.0)  #Product Price
      
                
        ])
    return np.array(ret)


# In[21]:


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)  #using model.predict as a transfer function

    return predictions

# Example simulation scenario with different policies

shipments_scenario = getScenarios(10000)   #calling function to generate data to run 10000 scenarios

predications=simulate_delivery_performance(shipments_scenario,LogReg) #running 10000 scenarios through transfer function


# In[22]:


#calculating probabilities of reaching on time versus late delivery

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# Scenario 2
# Market= Europe

# In[24]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #0
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 1,                    #14 Europe (fixing it at 1)
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 4), #21
                 random.uniform(-500.0, 500.0), #22
                 random.uniform(5.0, 2000.0), #23
                 random.randint(0, 49), #24
                 random.randint(1, 5), #25
                 random.randint(0, 22), #26
                 random.randint(0, 1), #27
                 random.uniform(8.0, 2000.0) #28
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,model)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# Scenario 3
# Market= Latin America

# In[25]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #0
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 1, #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 4), #21
                 random.uniform(-500.0, 500.0), #22
                 random.uniform(5.0, 2000.0), #23
                 random.randint(0, 49), #24
                 random.randint(1, 5), #25
                 random.randint(0, 22), #26
                 random.randint(0, 1), #27
                 random.uniform(8.0, 2000.0) #28
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# Scenario 4
# Market= Pacific Asia

# In[26]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #0
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 1, #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 4), #21
                 random.uniform(-500.0, 500.0), #22
                 random.uniform(5.0, 2000.0), #23
                 random.randint(0, 49), #24
                 random.randint(1, 5), #25
                 random.randint(0, 22), #26
                 random.randint(0, 1), #27
                 random.uniform(8.0, 2000.0) #28
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# Scenario 5
# Market= USCA

# In[27]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #0
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 1,                    #17  fixing it at 1
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 4), #21
                 random.uniform(-500.0, 500.0), #22
                 random.uniform(5.0, 2000.0), #23
                 random.randint(0, 49), #24
                 random.randint(1, 5), #25
                 random.randint(0, 22), #26
                 random.randint(0, 1), #27
                 random.uniform(8.0, 2000.0) #28
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# On the basis of sales per customer
# - 

# less than 500 sales

# In[30]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 1), #21
                 random.randint(0, 4), #22
                 random.uniform(-500.0, 500.0), #23
                 random.uniform(0, 500.0), #24
                 random.randint(0, 49), #25
                 random.randint(1, 5), #26
                 random.randint(0, 22), #27
                 random.randint(0, 1), #28
                 random.uniform(8.0, 2000.0) #29
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# between 500 to 1000

# In[31]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 1), #21
                 random.randint(0, 4), #22
                 random.uniform(-500.0, 500.0), #23
                 random.uniform(500.0, 1000.0), #24
                 random.randint(0, 49), #25
                 random.randint(1, 5), #26
                 random.randint(0, 22), #27
                 random.randint(0, 1), #28
                 random.uniform(8.0, 2000.0) #29
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# 1000 to 1500

# In[32]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 1), #21
                 random.randint(0, 4), #22
                 random.uniform(-500.0, 500.0), #23
                 random.uniform(1000.0, 1500.0), #24
                 random.randint(0, 49), #25
                 random.randint(1, 5), #26
                 random.randint(0, 22), #27
                 random.randint(0, 1), #28
                 random.uniform(8.0, 2000.0) #29
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# 1500 to 2000

# In[33]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 1), #21
                 random.randint(0, 4), #22
                 random.uniform(-500.0, 500.0), #23
                 random.uniform(1500.0, 2000.0), #24
                 random.randint(0, 49), #25
                 random.randint(1, 5), #26
                 random.randint(0, 22), #27
                 random.randint(0, 1), #28
                 random.uniform(8.0, 2000.0) #29
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# There could be several reasons for the decrease in the probability of a shipment reaching on time as the revenue increases:
# 
# Increased complexity: Higher revenue orders may involve more complex logistics, such as larger or specialized products, customized orders, or additional handling requirements. These complexities can introduce more variables and factors that can potentially cause delays in the delivery process.
# 
# Resource allocation: With higher revenue orders, the company may need to allocate more resources to fulfill them. This could include additional manpower, transportation, or other logistical resources. If the company is unable to effectively manage and allocate these resources, it may result in delays and lower on-time delivery rates.
# 
# Supply chain constraints: As revenue increases, the demand for products may also increase. This can put additional strain on the supply chain, including inventory management, production capacity, and shipping capabilities. If the supply chain is not adequately prepared to handle the increased demand, it can lead to delays and lower on-time delivery rates.
# 
# Customer expectations: Higher revenue orders may come with higher customer expectations and service-level agreements. Customers may have specific delivery timeframes or requirements for these orders, and meeting these expectations consistently can be more challenging. Failure to meet these expectations can result in lower on-time delivery rates.
# 

# On the basis of product Price
# - 

# low price 0-500

# In[48]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 1), #21
                 random.randint(0, 4), #22
                 random.uniform(-500.0, 500.0), #23
                 random.uniform(5.0, 2000.0), #24
                 random.randint(0, 49), #25
                 random.randint(1, 5), #26
                 random.randint(0, 22), #27
                 random.randint(0, 1), #28
                 random.uniform(0.0, 500.0) #29
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# low medium

# In[49]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 1), #21
                 random.randint(0, 4), #22
                 random.uniform(-500.0, 500.0), #23
                 random.uniform(5.0, 2000.0), #24
                 random.randint(0, 49), #25
                 random.randint(1, 5), #26
                 random.randint(0, 22), #27
                 random.randint(0, 1), #28
                 random.uniform(500, 1000.0) #29
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# medium high

# In[50]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 1), #21
                 random.randint(0, 4), #22
                 random.uniform(-500.0, 500.0), #23
                 random.uniform(5.0, 2000.0), #24
                 random.randint(0, 49), #25
                 random.randint(1, 5), #26
                 random.randint(0, 22), #27
                 random.randint(0, 1), #28
                 random.uniform(1000, 1500.0) #29
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# high price
# 1500-2000

# In[51]:


import random
import numpy as np
def getScenarios(numSamples):
    ret=[]
    for i in range(numSamples):
        ret.append([
                 random.randint(0, 1), #1
                 random.randint(0, 1), #2
                 random.randint(0, 1), #3
                 random.randint(0, 1), #4
                 random.randint(0, 1), #5
                 random.randint(0, 1), #6
                 random.randint(0, 1), #7
                 random.randint(0, 1), #8
                 random.randint(0, 1), #9
                 random.randint(0, 1), #10
                 random.randint(0, 1), #11
                 random.randint(0, 1), #12
                 random.randint(0, 1), #13
                 random.randint(0, 1), #14
                 random.randint(0, 1), #15
                 random.randint(0, 1), #16
                 random.randint(0, 1), #17
                 random.randint(0, 1), #18
                 random.randint(0, 1), #19
                 random.randint(0, 1), #20
                 random.randint(0, 1), #21
                 random.randint(0, 4), #22
                 random.uniform(-500.0, 500.0), #23
                 random.uniform(5.0, 2000.0), #24
                 random.randint(0, 49), #25
                 random.randint(1, 5), #26
                 random.randint(0, 22), #27
                 random.randint(0, 1), #28
                 random.uniform(1500.0, 2000.0) #29
      
                
        ])
    return np.array(ret)


import random
import numpy as np

# Function to simulate delivery performance based on different scenarios
def simulate_delivery_performance(scenario, model):
    # Initialize variables
    total_shipments= len(scenario)
    predictions=model.predict(scenario)

    return predictions

# Example simulation scenario with different policies
shipments_scenario = getScenarios(10000)  
predications=simulate_delivery_performance(shipments_scenario,LogReg)

vals,counts=np.unique(predications,return_counts=True)
print("reached on time",counts[0]/(counts[0]+counts[1]) )
print("reached late",counts[1]/(counts[0]+counts[1]) )


# the probablity of product reaching on time directly proportional to the price of the product.

# 
