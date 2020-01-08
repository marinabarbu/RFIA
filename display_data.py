import pandas as pd
import matplotlib.pyplot as plt

dataset2 = pd.read_excel(io="trainx.xlsx", sheet_name='Sheet1')
#print(dataset2)
x1i1 = dataset2.iloc[2:20,0].values
print(x1i1)

x2i1 = dataset2.iloc[2:20,1].values
print(x2i1)

x1i2 = dataset2.iloc[2:20,2].values
print(x1i2)

x2i2 = dataset2.iloc[2:20,3].values
print(x2i2)

plt.plot(x1i1,x2i1, 'r')
plt.plot(x1i2, x2i2, 'b')
plt.show()







