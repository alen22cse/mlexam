import pandas as pd 
data = pd.read_csv(r"C:\Users\HP\Downloads\training_data.csv")
print(data)
def find_s_algo(data):
    attributes = data.iloc[:,:-1].values
    target = data.iloc[:,-1].values
    hypothesis = None
    for i in range(len(target)):
        if target[i] == "Yes":
            hypothesis = attributes[i].copy()
            break
    if hypothesis is None:
        return "No postitive samples were found"
    for i in range(len(target)):
        if target[i] == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes[i][j]:
                    hypothesis[j] ="?"
    return hypothesis
print(find_s_algo(data))
