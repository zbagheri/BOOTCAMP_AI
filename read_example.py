"""   Created on Thu May  9 14:00:58 2024

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
df = pd.read_csv('example1.csv')

df2=list(df)
data = []
tag= df2[8]
temp1 = df[df2[8]][:100-1]   
plt.plot(temp1,label = tag+str( ' station1') )

    




