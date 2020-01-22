from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def get_nearest_neighbor(train, test_row):
    distances = list()
    for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    nearest_neighbor = distances[0][0]
    return nearest_neighbor

def predict_classification(train, test_row):
	nearest_neighbor = get_nearest_neighbor(train, test_row)
	return nearest_neighbor[-1]


def apply_nn(train_dataset):
    test_dataset_xlsx = pd.read_excel(io="train.xlsx", sheet_name='test')
    x1i1 = test_dataset_xlsx.iloc[2:20,0].values
    #print(x1i1)
    x2i1 = test_dataset_xlsx.iloc[2:20,1].values
    #print(x2i1)
    x1i2 = test_dataset_xlsx.iloc[2:20,2].values
    #print(x1i2)
    x2i2 = test_dataset_xlsx.iloc[2:20,3].values
    #print(x2i2)
    test_dataset = []
    for i in range(len(x1i1)):
        x = [x1i1[i], x2i1[i], 1]
        test_dataset.append(x)
        
    for i in range(len(x1i2)):
        x = [x1i2[i], x2i2[i], 2]
        test_dataset.append(x)

    correct_predictions = 0
    
    for x in test_dataset:
        print(get_nearest_neighbor(train_dataset,x))
        prediction = predict_classification(train_dataset, x)
        if prediction == x[-1]:
            correct_predictions += 1

    accuracy = correct_predictions/len(test_dataset)*100
    return accuracy
    

train_dataset_xlsx = pd.read_excel(io="train.xlsx", sheet_name='train')

x1i1 = train_dataset_xlsx.iloc[2:20,0].values
#print(x1i1)
x2i1 = train_dataset_xlsx.iloc[2:20,1].values
#print(x2i1)
x1i2 = train_dataset_xlsx.iloc[2:20,2].values
#print(x1i2)
x2i2 = train_dataset_xlsx.iloc[2:20,3].values
    
plt.plot(x1i1,x2i1, 'o', color='red', label="Clasa 1")
plt.plot(x1i2, x2i2, '*',color='blue', label="Clasa 2")
plt.show()

train_dataset = []

for i in range(len(x1i1)):
    x = [x1i1[i], x2i1[i], 1]
    train_dataset.append(x)
        
for i in range(len(x1i2)):
    x = [x1i2[i], x2i2[i], 2]
    train_dataset.append(x)
    
accuracy = apply_nn(train_dataset)
print(accuracy)
