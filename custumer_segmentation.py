import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
dataframe = pd.read_csv("Mall_Customers.csv",encoding="latin1")
'''
==================================================
Algorithm: KNN
==================================================
Best Parameters: {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 3, 'p': 2, 'weights': 'distance'}
Best CV Score: 0.9119
Train Score: 100.00%
Test Score: 87.23%
==================================================
'''
# print(dataframe.head(10))
data_processed_0 = dataframe.drop(columns=["Gender"],axis=1)
group = {"Group 0": "Low income, low spending",
"Group 1": "High income, low spending",
"Group 2": "Low income, high spending",
"Group 3": "High income, high spending",
"Group 4": "Middle income, moderate spending",
"Group 5": "Senior customers",
"Group 6": "Young high spenders"}
for key,value in group.items():
    print(key,value)
ohe = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
def preprocess(dataframe):
    n_m_f = ohe.fit_transform(dataframe[["Gender"]])
    n_f = pd.DataFrame(n_m_f,columns=ohe.get_feature_names_out(["Gender"]))
    data_processed = pd.concat([data_processed_0,n_f],axis=1)
    return data_processed
def cluster_value(data_processed):
    wscc = []
    for i in range(2,21):
        kmr = KMeans(n_clusters=i,init="k-means++")
        kmr.fit(data_processed)
        wscc.append(kmr.inertia_)
    plt.plot([i for i in range(2,21)],wscc,color = "royalblue",marker="o",markerfacecolor="black")
    plt.xticks([i for i in range(2,21)])
    plt.xlabel("Number of cluster")
    plt.ylabel("Wscc")
    plt.grid(True)
    # plt.savefig("Elbow_graph_2")
    plt.show()
    return wscc
def clustring(data_processed):
    kmm = KMeans(n_clusters=7,init="k-means++")
    data_processed["Grouping"]=kmm.fit_predict(data_processed)
    data_processed.sort_values("Grouping",inplace=True)
    sns.pairplot(data_processed,hue="Grouping")
    # plt.savefig("Cluster_2")
    plt.show()
    data_processed.to_csv("data.csv")
    return data_processed
def graph(data_processed):
    sns.pairplot(data_processed)
    plt.show()
def training(data_processed):
    x = data_processed.iloc[:,:-1]
    y = data_processed["Grouping"]
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.23,random_state=42)
    knc = KNeighborsClassifier(algorithm="auto",leaf_size=10,n_neighbors=3,p=2,weights="distance")
    knc.fit(X_train,y_train)
    y_pred = knc.predict(X_test)
    print("Train Score:", knc.score(X_train, y_train) * 100)
    print("Test Score:", accuracy_score(y_test, y_pred) * 100)
    return knc  
def work(knc,ohe,data_column):
    try:
        cus_id = int(input("Enter the Custumer id:  "))
        Age = int(input("Enter Your Age:  "))
        annual_inc = int(input("Enter Your Annual income in (k$): "))
        sp_score = int(input("Enter Your Spending Score (1-100):  "))
        gender = input("Enter your gender (Male or female) : ").strip()
        gender_encoded = ohe.transform([[gender]]).flatten()
        input_data = [cus_id,Age,annual_inc,sp_score] + list(gender_encoded)
        input_df = pd.DataFrame([input_data],columns=data_column)
        pred = knc.predict(input_df)[0]
        print(f"The predicted group is: Group {pred} → {group.get(f'Group {pred}', 'Unknown')}")
    except Exception as e:
        print("The error is -",e)         
        
def main():
    print("="*50)
    print("Customer segmentation")
    print("="*50)
    
    data_processed = preprocess(dataframe)
    wscc = cluster_value(data_processed)
    print("The Prediction Of the Data is ")
    data_processed = clustring(data_processed)
    knc = training(data_processed)
    print("The Prediction is ready......")
    work(knc,ohe,data_processed.columns[:-1])
if __name__ == "__main__":
    main()