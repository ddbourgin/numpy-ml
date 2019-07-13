import numpy as np
from wrappers import AlphaDropout

input = np.array([1,2,3,4,6,8,1,2,4,5])
#print(input)
ad = AlphaDropout(Softmax(),0.5)
y = ad.forward(input)
print(y)