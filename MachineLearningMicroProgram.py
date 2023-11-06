#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')
df=pd.read_csv("./CropRecommendation.csv")
df.head()


# In[2]:


df.describe


# In[3]:


df.isnull().sum()


# In[4]:


n=df.N
p=df.P
k=df.K
temp=df.temperature
hum=df.humidity
ph=df.ph
rain=df.rainfall


# In[5]:


plt.scatter(n,df.label)
plt.xlabel('Nitrogen content in soil')
plt.ylabel('Names of crops')


# In[6]:


plt.scatter(p,df.label)
plt.xlabel('Phosphorus content in soil')
plt.ylabel('Names of crops')


# In[7]:


plt.scatter(k,df.label)
plt.xlabel('Potassium content in soil')
plt.ylabel('Names of crops')


# In[8]:


plt.scatter(temp,df.label)
plt.xlabel('Temperature in degree Celsius')
plt.ylabel('Names of crops')


# In[9]:


plt.scatter(hum,df.label)
plt.xlabel('Relative Humidity in percentage')
plt.ylabel('Names of crops')


# In[10]:


plt.scatter(ph,df.label)
plt.xlabel('pH of soil')
plt.ylabel('Names of crops')


# In[11]:


plt.scatter(rain,df.label)
plt.xlabel('Annual rainfall in mm')
plt.ylabel('Names of crops')


# In[12]:


df['label'].unique()


# In[13]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Crop_num=le.fit_transform(df.label)
Crop_num=pd.DataFrame(Crop_num ,columns=['Crop_number'])
Crop_num.head()


# In[14]:


df3=pd.concat([df,Crop_num],axis=1)


# In[15]:


df3['Crop_number'].unique()


# In[16]:


LB=pd.DataFrame(df3['label'].unique(),columns=['Crop Name'])
RN=pd.DataFrame(df3['Crop_number'].unique(),columns=['Crop_Number'])
Rep_Numb=pd.concat([LB,RN],axis=1)                          #Creating a reference list for identifying crops 
Rep_Numb


# In[17]:


df2=df.drop(['label'],axis=1) 
df2.head()


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df2,Crop_num,train_size=0.8)


# In[19]:


from sklearn import tree                       #Decision Tree Algorithm
dt=tree.DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[20]:


print("accuracy score: ",dt.score(x_test,y_test)*100,"%")


# In[21]:


dt.predict([[111,29,31,26.0596,52.31099,6.136287,161.6269]])


# In[22]:


from sklearn.ensemble import RandomForestClassifier         #Random Forest Algorithm
RF=RandomForestClassifier(n_estimators=5)
RF.fit(x_train,y_train)


# In[23]:


print("accuracy score: ",RF.score(x_test,y_test)*100,"%")


# In[24]:


from sklearn.neighbors import KNeighborsClassifier       #K Nearest Neighbors Algorithm
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)


# In[25]:


print("accuracy score: ",knn.score(x_test,y_test)*100,"%")


# In[26]:


arr=np.array([9,66,21,30.11812084,34.13307843,5.719889876,157.0858232])
#pred=dt.predict([arr])
pred=RF.predict([arr])
#pred=knn.predict([arr])


# In[27]:


match pred:
    case 0:
        print("Apple")
    case 1:
        print("Banana")
    case 2:
        print("Blackgram")
    case 3:
        print("Chickpea")
    case 4:
        print("coconut")
    case 5:
        print("coffee")
    case 6:
        print("cotton")
    case 7:
        print("Grapes")
    case 8:
        print("Jute")
    case 9:
        print("Kidneybeans")
    case 10:
        print("Lentil")
    case 11:
        print("Maize")
    case 12:
        print("Mango")
    case 13:
        print("Mothbean")
    case 14:
        print("Mungbean")
    case 15:
        print("Muskmelon")
    case 16:
        print("Orange")
    case 17:
        print("Papaya")
    case 18:
        print("Pegionpeas")
    case 19:
        print("Pomegranate")
    case 20:
        print("Rice")
    case 21:
        print("Watermelon")


# In[28]:


import pickle
dec_tr=open("DT_Model","wb")
lm=pickle.dump(dt,dec_tr)


# In[29]:


rndm_frst=open("RF_Model","wb")
lm=pickle.dump(RF,rndm_frst)


# In[30]:


kn_n=open("KNN_Model","wb")
lm=pickle.dump(knn,kn_n)

