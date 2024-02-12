from dlroms import*

def downloadROM(mutrain, utrain):
  """ See below for the definition and training of such ROM (POD-NN model) """
  basis, svalues = POD(utrain, k = 100)
  V = basis.T
  coeffs_train = utrain.mm(V)

  # Input-output normalization
  mu_std = mutrain.var(axis = 0).sqrt().unsqueeze(0)
  mu_mean = mutrain.mean(axis = 0).unsqueeze(0)
  
  c_std = coeffs_train.var(axis = 0).sqrt().unsqueeze(0)
  c_mean = coeffs_train.mean(axis = 0).unsqueeze(0)
  
  # Architecture setup
  from dlroms.roms import DFNN
  phi = DFNN(Dense(5, 100) + Dense(100, 200) + Dense(200, V.shape[1], activation = None))

  import gdown
  gdown.download(id = "1xGKJGeAes3ZhdkHKrEXH8GY0jcJwhmnP", output = "DTphi.npz", quiet=False)
  phi.load("DTphi.npz")
  phi.freeze()
  from dlroms.dnns import ReLU
  
  ### Online phase
  
  def ROM(m):
    return ReLU((c_std*phi((m-mu_mean)/mu_std)+c_mean).mm(basis))

  return ROM

""" Implementation of the NN model + training

from dlroms import*

# POD basis and POD projection
ntrain = int(len(u)*0.8)
basis, svalues = POD(u[:ntrain], k = 100)
V = basis.T
coeffs = u.mm(V)

# Input-output normalization
mu_std = mu[:ntrain].var(axis = 0).sqrt().unsqueeze(0)
mu_mean = mu[:ntrain].mean(axis = 0).unsqueeze(0)

c_std = coeffs[:ntrain].var(axis = 0).sqrt().unsqueeze(0)
c_mean = coeffs[:ntrain].mean(axis = 0).unsqueeze(0)

# Architecture setup
from dlroms.roms import DFNN
p = 5
phi = DFNN(Dense(p, 100) + Dense(100, 200) + Dense(200, V.shape[1], activation = None))
phi.He()

# Training phase: loss terms weighted according to singular values (captures POD coeffs. hierarchy without changing output scale)
normweights = svalues/svalues[0]

def norm(c, squared = False):
  value = (c*normweights).pow(2).sum(axis = -1)
  return value if squared else value.sqrt()

input = (mu - mu_mean)/mu_std
output = (coeffs - c_mean)/c_std

phi.train(input, output, ntrain = ntrain, epochs = 1000, loss = mse(norm))
phi.freeze()
"""
  
