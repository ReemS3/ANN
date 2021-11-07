import numpy as np

""" Class for functions that are used in multiple classes. """
def sigmoid(x):
  return 1.0/(1+np.exp(-x))

def sigmoidprime(x): 
  """Assuming x has been computed with sigmoid first"""
  return x * (1 - x)
