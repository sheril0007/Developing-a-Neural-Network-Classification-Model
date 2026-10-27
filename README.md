# Developing a Neural Network Classification Model

### Name: NITHISH KUMAR S

### Register Number:212224230190

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
<img width="1287" height="396" alt="image" src="https://github.com/user-attachments/assets/b0bfe53d-0fe1-4d2a-a2a9-fb1fc38cb706" />

```
df=df.drop(columns=["ID"])
df
```

<img width="1221" height="427" alt="image" src="https://github.com/user-attachments/assets/bbb25a7c-2267-44e6-8bdf-62b7e0c3d220" />

```
df.columns
```
<img width="740" height="165" alt="image" src="https://github.com/user-attachments/assets/4ea3ea89-36fc-47c6-955e-9947399346d5" />

```
df.columns
```
<img width="810" height="130" alt="image" src="https://github.com/user-attachments/assets/a1ab6b8e-b2d6-4823-a288-7e02283b22cf" />

```
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
df
```
<img width="1201" height="415" alt="image" src="https://github.com/user-attachments/assets/0dcd5ffa-a6e7-4e89-8238-6d492c58312a" />

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
<img width="1242" height="443" alt="image" src="https://github.com/user-attachments/assets/1cf617d4-2342-4215-877e-79c87927127c" />


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
<img width="422" height="262" alt="image" src="https://github.com/user-attachments/assets/bfb80a69-2594-482d-beac-90b20e94aa3a" />

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
<img width="1747" height="80" alt="image" src="https://github.com/user-attachments/assets/e9bdd7ce-4d17-49ba-8788-b32490a2f015" />


```
accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=['A','B','C','D'])
print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)
print("classification_report:\n",cl_report)
```
<img width="696" height="440" alt="image" src="https://github.com/user-attachments/assets/f2d226d4-8a4a-4713-ba39-7871ed6433e3" />


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

<img width="672" height="576" alt="image" src="https://github.com/user-attachments/assets/9ae61567-6a8f-4e6c-b079-f2ec0053d224" />



### Dataset Information
<img width="1331" height="432" alt="image" src="https://github.com/user-attachments/assets/5bbb1ad9-c1d8-4eb8-941d-5cda7275bea2" />


### OUTPUT

## Confusion Matrix

<img width="672" height="576" alt="image" src="https://github.com/user-attachments/assets/9ae61567-6a8f-4e6c-b079-f2ec0053d224" />

## Classification Report
<img width="696" height="440" alt="image" src="https://github.com/user-attachments/assets/f2d226d4-8a4a-4713-ba39-7871ed6433e3" />


## RESULT
This program has been executed successfully.
