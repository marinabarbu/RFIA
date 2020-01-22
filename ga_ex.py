import numpy
import nearest_neighbor as nn
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt

def crossover(parent_1, parent_2):
    offspring = parent_1
    for i in range(int(len(offspring)/2), len(offspring)):
        offspring[i] = parent_2[i]
    return offspring

def mutation(offspring_crossover, p):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        if numpy.random.rand() > 1-p:
            offspring_crossover[idx] = int(not offspring_crossover[idx])
    return offspring_crossover


population = []
for i in range(8):
    population.append(numpy.random.randint(2, size=36))
population = numpy.array(population)
#print(population.shape)

train_dataset_xlsx = pd.read_excel(io="train.xlsx", sheet_name='train')
x1i1 = train_dataset_xlsx.iloc[2:20,0].values
x2i1 = train_dataset_xlsx.iloc[2:20,1].values
x1i2 = train_dataset_xlsx.iloc[2:20,2].values
x2i2 = train_dataset_xlsx.iloc[2:20,3].values

num_generations = 300
acc_hist = []
max_acc = 0

for generation in range(num_generations):
    print("Generation : ", generation)
    chr_and_acc_list = []
    for chr in population:
        new_x1i1, new_x2i1, new_x1i2, new_x2i2 = [], [], [], []
        # Measing the fitness of each chromosome in the population.
        for i in range(18):
            if chr[i] == 1:
                new_x1i1.append(x1i1[i])
                new_x2i1.append(x2i1[i])
                
        for i in range(18,36):
            if chr[i] == 1:
                new_x1i2.append(x1i2[i-18])
                new_x2i2.append(x2i2[i-18])
                
        train_dataset = []
        for i in range(len(new_x1i1)):
            x = [new_x1i1[i], new_x2i1[i], 1]
            train_dataset.append(x)
            
        for i in range(len(new_x1i2)):
            x = [new_x1i2[i], new_x2i2[i], 2]
            train_dataset.append(x)
            
        accuracy = nn.apply_nn(train_dataset)
        
        chr_and_acc_list.append([chr, accuracy])

    chr_and_acc_list = sorted(chr_and_acc_list, key=itemgetter(1))
    if chr_and_acc_list[-1][1] > max_acc:
        max_acc = chr_and_acc_list[-1][1]
    acc_hist.append(max_acc)
    print("Accuracy: ", max_acc)
    #print(chr_and_acc_list)

    parent_1, parent_2 = chr_and_acc_list[-1][0], chr_and_acc_list[-2][0]
    
    #print(parent_1)
    #print(parent_2)
    
    offspring = crossover(parent_1, parent_2)
    population[0] = offspring
    #print(offspring)
    #offspring_mutation = mutation(offspring, p=0.05)
    #print(offspring_mutation)
    population = [mutation(chr, p=0.2) for chr in population]
    #population[0]= offspring_mutation

plt.plot( [i for i in range(num_generations)],acc_hist)
plt.xlabel("Generations")
plt.ylabel("Accuracy")
plt.show()