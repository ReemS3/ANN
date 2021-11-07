import numpy as np

# warnings.filterwarnings('ignore')
def sigmoid(x):
  return 1.0/(1+np.exp(-x))

def sigmoidprime(x): 
  return x * (1 - x)