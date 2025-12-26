import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a=np.arange(100).reshape(25,4)
print(a)
print(a.shape)
print(a.ndim)
print(a.size)
df=pd.date_range("2025/12/24",periods=4)
print(df)
df2=pd.DataFrame(np.random.randint(20,size=(4,5)),index=df)
print(df2)
print(df2.describe())
df2.plot(kind="scatter", x=df2.index ,y="[0,1,2,3,4]")
plt.show()