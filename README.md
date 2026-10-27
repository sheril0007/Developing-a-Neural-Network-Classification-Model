# Developing a Neural Network Classification Model

### Name: SHERIL.P

### Register Number:212225230262

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load and Preprocess the Dataset:

Import the required libraries, load the dataset, handle missing values, encode categorical variables, and normalize the features.

### STEP 2: 

Split the Dataset:

Divide the dataset into training and testing sets.

### STEP 3: 

Build the Neural Network:

Create a Sequential model with an input layer, hidden layers using ReLU activation, and an output layer with Softmax activation for the four customer segments.

### STEP 4: 

Train the Model:

Compile the model using the Adam optimizer and categorical cross-entropy loss function, then train it using the training data.

### STEP 5: 

Evaluate and Predict:

Evaluate the model on the test dataset, measure its accuracy, and use it to predict the customer segment (A, B, C, or D) for new customers.







## PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
```
df=pd.read_csv("C:\\Users\\varal\\Downloads\\Deep Learning\\Deep Learning\\customers.csv")
df
```
<img width="1331" height="457" alt="image" src="https://github.com/user-attachments/assets/4edae1d4-156f-44f9-82a5-d31ab39d4603" />

```
df=df.drop(columns=["ID"])
df
```

<img width="1291" height="462" alt="image" src="https://github.com/user-attachments/assets/28224ca4-e0c4-402c-a6ce-a07f523bb2ec" />


```
df.columns
```
<img width="932" height="122" alt="image" src="https://github.com/user-attachments/assets/1a4bc643-1558-47f1-8f38-b1e051f8dcaf" />


```
```
```
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
df
```
<img width="1438" height="460" alt="image" src="https://github.com/user-attachments/assets/ec1ec6d5-4d47-473b-99ac-a8c3146ac652" />


```
cat_columns=['Gender','Ever_Married', 'Graduated', 'Profession',
    'Spending_Score', 'Var_1']
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])
```

```
lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])
df
```
<img width="1282" height="455" alt="image" src="https://github.com/user-attachments/assets/080fd711-675f-42ed-9948-6076ae8fbbf5" />



```
x=df.drop(columns="Segmentation")
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)
```

```
scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)
```

```
xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)
```

```
tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)
```

```
class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,8)
        self.l4=nn.Linear(8,4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x
```

```
model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
```

```
epochs=100
for i in range(epochs):
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b.long())
        loss.backward()
        op.step()
    if i%10==0:
        print(f"Loss:{i}/{epochs}",loss.item())
```
<img width="592" height="266" alt="image" src="https://github.com/user-attachments/assets/d16aeaf2-6129-4598-92e9-f92f9edd3287" />


```
pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    act.extend(yst.numpy())
    print(act,pre)
```
<img width="1360" height="66" alt="image" src="https://github.com/user-attachments/assets/7bb33ce4-8b4b-4b02-aa71-ecc83749efa8" />



```
accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=['A','B','C','D'])
print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)
print("classification_report:\n",cl_report)
```
<img width="880" height="430" alt="image" src="https://github.com/user-attachments/assets/b7383a50-708c-4ec8-95b0-2fb50d4c6ebe" />



```
import seaborn as sns
import matplotlib.pyplot as plt
xl=['A','B','C','D']
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=xl, yticklabels=xl,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```

<img width="998" height="577" alt="image" src="https://github.com/user-attachments/assets/667b96f7-f1bf-4e03-8af8-08fea6dc73d8" />




### Dataset Information
<img width="1362" height="471" alt="image" src="https://github.com/user-attachments/assets/64dddf1f-93b1-4b09-9f2d-c4948b199eed" />



### OUTPUT

## Confusion Matrix

<img width="1058" height="576" alt="image" src="https://github.com/user-attachments/assets/f552e9eb-267d-446e-bea0-abf99d8d82a3" />


## Classification Report
<img width="962" height="428" alt="image" src="https://github.com/user-attachments/assets/3a2e0781-31e8-4e58-bf45-9f35bcc4883d" />



## RESULT
This program has been executed successfully.
