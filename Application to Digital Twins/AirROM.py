from dlroms import*

def downloadROM(mutrain, utrain):
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
  
  
