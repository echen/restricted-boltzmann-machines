import numpy as np

class RBM:

  def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.learning_rate = learning_rate

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a Gaussian distribution with mean 0.1.
    self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)    
    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)
    
  def daydream(self, num_samples):
    """
    Randomly initialize the visible units, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    
    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """
        
    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.        
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the hidden bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = logistic(visible_activations)
      samples[i,:] = visible_probs

    # Ignore the visible bias units (the first column), since they're always set to 1.
    return samples[:,1:]

  def run(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a hidden bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert hidden bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the hidden bias unit to 1.
    hidden_states[:,0] = 1
  
    return hidden_states

  def train(self, data, max_epochs = 50):
    """
    Train the machine.
    
    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      pos_products = np.dot(data.T, pos_hidden_probs)

      # Reconstruct the visible units and sample again from the hidden units. 
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      neg_products = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += self.learning_rate * ((pos_products - neg_products) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      print "Epoch %s: error is %s\n" % (epoch, error)
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))      

if __name__ == '__main__':
  r = RBM(num_visible = 10, num_hidden = 2)
  data = np.array([[1,1,1,1,1,0,0,0,0,0],[0,0,1,0,0,1,1,1,1,0],[1,1,1,1,0,0,0,0,0,0],[0,0,1,0,0,0,1,1,1,0], [0,0,1,0,0,0,1,1,1,0],[1,1,1,1,1,0,0,0,0,0]])
  r.train(data, max_epochs = 1000)
  print r.weights