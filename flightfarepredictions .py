#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


pd.set_option('display.max_columns',None)


# In[3]:


df=pd.read_excel(r'C:\Users\pc\Desktop\ML\flightfare\Data_Train.xlsx')


# In[4]:


df


# # EDA

# In[5]:


#date_of_journy
import datetime
df["Journey_day"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day


# In[6]:


df["Journey_month"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.month


# In[7]:


df['Dep_hour']=pd.to_datetime(df['Dep_Time']).dt.hour


# In[8]:


df['Dep_min']=pd.to_datetime(df['Dep_Time']).dt.minute


# In[9]:


df


# In[10]:


duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour


# In[11]:


duration


# In[12]:


duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[13]:


df['duration_hours']=duration_hours
df['duration_mins']=duration_mins


# In[14]:


df.head()


# In[15]:


df['arrival_hour']=pd.to_datetime(df['Arrival_Time']).dt.hour
#train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour


# In[16]:


df['arrival_min']=pd.to_datetime(df['Arrival_Time']).dt.minute


# In[17]:


df.head()


# In[18]:


df.drop(['Dep_Time','Arrival_Time','Duration'],axis=1,inplace=True)


# In[19]:


df.head()


# In[20]:


df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[21]:


df.head()


# In[ ]:





# In[22]:


df['Source'].unique()


# In[23]:


df['Destination'].unique()


# In[24]:


df.isnull().sum()


# In[25]:


df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[26]:


df.isnull().sum()


# In[27]:


df['Total_Stops'].value_counts()


# In[28]:


df[df['Total_Stops'].isnull()]


# In[29]:


df.drop(9039,inplace=True)


# In[30]:


df


# In[31]:


df['Total_Stops']=df['Total_Stops'].astype(np.int64)


# In[32]:


df.head()


# In[33]:


df.drop(['Date_of_Journey'],axis=1,inplace=True)


# In[34]:



for feature in df.columns:
    data=df.copy()
    data.groupby(feature)['Price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title(feature)
    plt.show()


# In[35]:


df.groupby('Airline')['Price'].median().plot.bar()


# In[ ]:





# In[ ]:





# In[37]:


# As Airline is Nominal Categorical data we will perform OneHotEncoding

Airline = df[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[38]:


# As Source is Nominal Categorical data we will perform OneHotEncoding

Source = df[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[39]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination = df[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[40]:


data_train = pd.concat([df, Airline, Source, Destination], axis = 1)


# In[41]:


data_train


# In[42]:


data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[43]:


data_train


# 
# # TEST data
# 

# In[47]:


test_data=pd.read_excel(r'C:\Users\pc\Desktop\ML\flightfare\Data_test.xlsx')


# In[48]:


pd.set_option('display.max_columns',None)


# In[49]:


# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[50]:


data_test.head()


# # feature selection

# In[52]:


data_train.shape


# In[53]:


data_train.columns


# In[55]:


X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'duration_hours', 'duration_mins', 'arrival_hour',
       'arrival_min', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]


# In[56]:


y=data_train['Price']


# In[60]:



plt.figure(figsize = (25,25))
sns.heatmap(data_train.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# In[61]:


# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[62]:


print(selection.feature_importances_)


# In[63]:


#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[66]:


X


# In[67]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[68]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)


# In[69]:


y_pred = reg_rf.predict(X_test)


# In[70]:


reg_rf.score(X_train, y_train)


# In[71]:


reg_rf.score(X_test, y_test)


# In[72]:


sns.distplot(y_test-y_pred)
plt.show()


# In[73]:


plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[74]:


from sklearn import metrics


# In[75]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[76]:


# RMSE/(max(DV)-min(DV))

2090.5509/(max(y)-min(y))


# In[77]:


metrics.r2_score(y_test, y_pred)


# In[81]:


import pickle
# open a file, where you ant to store the data
file = open(r'C:\Users\pc\Desktop\ML\flightfare\flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(reg_rf, file)


# In[84]:


model = open(r'C:\Users\pc\Desktop\ML\flightfare\flight_rf.pkl','rb')
forest = pickle.load(model)


# In[ ]:




