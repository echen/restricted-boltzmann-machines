(This is a rambly tutorial in progress.)

Suppose you have a set of movies, and you ask each of your friends whether they want to watch it (getting a binary yes/no answer). Let's see how we can use a Restricted Boltzmann Machine to learn latent movie groups that will help explain your friends' preferences.

# The Movie Setup

To be concrete, let's use the following 10 movies in our set:

* LOTR 1: The Fellowship of the Ring
* LOTR 2: The Two Towers
* LOTR 3: The Return of the King
* Harry Potter and the Sorcerer's Stone
* Twilight
* Titanic
* Gladiator
* A Beautiful Mind
* Slumdog Millionaire
* Justin Bieber: Never Say Never

Notice that there are some natural groups behind the movies:

* **Science fiction and Fantasy** (LOTR 1-3, Harry Potter, Twilight)
* **Oscar Best Picture Winners** (LOTR 3, Titanic, Gladiator, A Beautiful Mind, Slumdog Millionaire)
* **Tween movies** (Harry Potter, Twilight, Justin Bieber)
* **Russell Crowe movies** (Gladiator, A Beautiful Mind)
* and so on.

Our goal is to discover clusters like these that can explain why your friends want to watch what they do.

# The User Setup

Again, to be concrete, let's say your friends show the following preferences:

* Alice: loves science fiction and fantasy. She wants to watch LOTR 1-3, HP, and Twilight.
* Bob: only watches critically acclaimed movies. He chooses all the Oscar winners.
* Carol: loves SF&F, but refuses to watch Twilight. She chooses LOTR 1-3 and HP.
* David: watches all the Oscar winners except Titanic.
* Eve: is close friends with Alice, so she also chooses all the SF&F movies.
* Fred: watches all the Oscar winners, plus Twilight because of his girlfriend.

With this setup, let's now talk about how to stick movies and preferences into an RBM.

# The Restricted Boltzmann Machine Setup

A Restricted Boltzmann Machine is essentially a two-layer neural network. One layer consists of *visible* units (representing the movies a user wants to watch, in our example), each of which can be either in an active 1 state (meaning the user wants to watch it) or in an inactive 0 state (meaning the user does not want to watch it); the other layer consists of *hidden* units (latent groups underlying our movies), which again can be either in an active or inactive state. Since we want to be able to explain movie preferences in terms of these hidden groups, we connect each movie to all the hidden units. We also add one *bias* unit, whose state is always fixed at 1, and connect it to all the visible and all the hidden units.

In our example, we have 10 visible units corresponding to movies, and let's say we have 2 hidden units that we'lll use to explain our friends' preferences. (In our example, our friends seemed to make their decisions based primarily on whether a movie was a SF&F movie or an Oscar winner, so we hope that these will be the two groups that we learn.)

[![Movie RBM](http://dl.dropbox.com/u/10506/blog/rbms/movies-rbm.png)](http://dl.dropbox.com/u/10506/blog/rbms/movies-rbm.png)

(Note, crucially, that none of the movies is connected to any other movie, and none of the hidden groups is connected to any other hidden group. This is why we call this network a *Restricted* Boltzmann Machine; this restriction is important because it makes learning much easier. If, instead, every unit were connected to every other, we would have a full Boltzmann Machine.)

(Also, does the graphical model look like [anything else](http://en.wikipedia.org/wiki/Factor_analysis) to you? Another way of thinking about RBMs is to interpret them as performing binary factor analysis.)

So what does this all mean? Suppose, for the moment, that we have assigned weights to all the edges: 

* All the SF&F movies have a strong positive weight to the first hidden unit (which we can interpret as the "SF&F unit").
* All the Oscar winners have a strong positive weight to the second hidden unit (which we can interpret as the "Oscar winners unit").
* LOTR 3 has a strong positive weight to both units (since it's both SF&F and an Oscar winner).
* Units that are relatively hard to turn on (e.g., relatively unpopular movies) are connected to the bias units with negative weights; units that are relatively easy to turn on (e.g., relatively popular movies) are connected to the bias units with positive weights.

Thus, the network has a natural interpretation where hidden units explain why movie units are on.

Now, given a user Jack's 10 movie preferences, let's see how we can use our RBM to model them.

* First, set each movie unit to 1 or 0 depending on whether Jack wants to watch the movie or not. (For simplicity, I've dropped some of the movies in the pictures.)

[![Movie RBM Initialized](http://dl.dropbox.com/u/10506/blog/rbms/movie-rbm-initialized.png)](http://dl.dropbox.com/u/10506/blog/rbms/movie-rbm-initialized.png)

* Next, have all the (visible and bias) units send a message to the hidden units, where the message from unit $i$ to hidden unit $j$ is the product of $x_i$ ($i$'s state) and $w_{ij}$ (the weight of the edge between $i$ and $j$).

[![Message to Hidden](http://dl.dropbox.com/u/10506/blog/rbms/message-to-hidden.png)](http://dl.dropbox.com/u/10506/blog/rbms/message-to-hidden.png)

* For example, if the LOTR 1 unit is active, and it's connected to the SF&F unit with a weight of 0.95, it sends a message of $1 * 0.95$ to the SF&F unit.
* Each hidden unit now computes the sum of all the messages it received. This is called the **activation energy**. If this sum is positive, the hidden unit turns on with high probability; otherwise, the hidden unit turns off with high probability. (More precisely, let $a_j = \sum w_{ij} x_i$ be the activation energy of the $j$th hidden unit, where the sum ranges over the visible units. Then we turn the $j$th hidden unit on with probability $\sigma(a_j)$, where $\sigma(x) = 1/(1 + exp(-x))$ is the logistic function.) We can interpret the new states of the hidden units as explanations for Jack's choices.

[![Hidden Activation](http://dl.dropbox.com/u/10506/blog/rbms/hidden-activation.png)](http://dl.dropbox.com/u/10506/blog/rbms/hidden-activation.png)

* Note that this updating rule makes sense: if Jack chooses to watch a lot of the SF&F movies, then since they're all connected to the SF&F unit with a large positive weight, they'll try to turn the unit on. Conversely, if Jack doesn't want to watch any of the SF&F movies, the SF&F unit will have a low activation energy, so it will probably remain off.
* But note also that even if Jack chooses to watch a lot of the SF&F movies, so that the SF&F unit has a high activation energy, this doesn't guarantee that the SF&F unit will actually turn on, because of the randomness involved in activation. This makes a bit of sense: if Jack watches a lot of SF&F movies, then it's very likely that it's because he likes SF&F in general, but there's a small chance that he just happened to watch those movies for other reasons.
* After updating our hidden units to their new states, we can also use the same process to update our visible units, to give us a **reconstruction** of our original data. We can interpret this as "correcting" Jack's original movie choices. For example, suppose Jack told us originally that he wanted to watch LOTR 1-3 and Twilight, but not Harry Potter. In setting the hidden units, it's likely the SF&F unit was turned on (since Jack told us he wanted to watch 4/5 of the SF&F movies), and so it will now try to turn the Harry Potter unit on in the reconstruction (which makes sense, since if Jack wants to watch LOTR 1-3 and Twilight, we might suspect he would want to watch Harry Potter as well).

[![Reconstruction](http://dl.dropbox.com/u/10506/blog/rbms/reconstruction.png)](http://dl.dropbox.com/u/10506/blog/rbms/reconstruction.png)

Now that we've seen a rough example of an RBM in action, let's talk about how to actually learn the edge weights in our network.

# Learning Weights

Here's how we can use our training examples to learn the edge weights in our Restricted Boltzmann Machine. In each epoch, do the following:

* Take a training example (a set of 10 movie preferences). Set the states of the visible units to these preferences.
* Next, update the states of the hidden units: for the $j$th hidden unit, compute its activation energy $a_j = \sum_i w_{ij} x_i$ (where the summation runs over the the visible units and the bias unit, and $x_i$ is the binary state of the $i$th unit), and set $x_j$ to +1 with probability $\sigma(a_j)$ and to 0 with probability $1 - \sigma(a_j)$ (where $\sigma(\cdot)$ is the logistic function). Then for each edge $e_{ij}$, set $Positive(e_{ij}) = x_i * x_j$.
* Now reconstruct the visible units in a similar manner: for each visible unit, compute its activation energy $a_i$, and update its state. Then update the hidden units again, and compute $Negative(e_{ij}) = x_i * x_j$ for each edge.
* Update the weight of each edge $e_{ij}$ by $w_{ij} = w_{ij} + L * (Positive(e_{ij}) - Negative(e_{ij}))$, where $L$ is a learning rate.
* Repeat over all training examples.

Continue doing this until the network converges (i.e., the error between the training examples and their reconstructions falls below some threshold) or we reach some maximum number of epochs.

Why does this update rule make sense? Note that 

* $Positive(e_{ij})$ measures the association between the $i$th and $j$th unit that we *want* the network to learn from our training examples;
* $Negative(e_{ij})$ measures the association that the network itself generates (or "daydreams" about) when no units are fixed. 

So by adding $Positive(e_{ij}) - Negative(e_{ij})$ to each edge weight, we're helping the network's daydreams match the reality of our training examples.

(You may hear this update rule called "contrastive divergence", which is basically a fancy term for "approximate gradient descent".)

# Some Optimizations

The weight-learning algorithm I described above is pretty simple, so let's talk about some modifications that you might see in practice:

* Above, $Negative(e_{ij})$ was determined by taking the product of the $i$th and $j$th units after *one* reconstruction. We could also take the product after some larger number of reconstructions (where a reconstruction consists of updating the visible units again, followed by updating the hidden units); this is slower, but describes the network's daydreams more accurately.
* Instead of using $Positive(e_{ij}))=x_i * x_j$, where $x_i$ and $x_j$ are the binary *states* of the units, we could also let $x_i$ and/or $x_j$ be the activation *probabilities*.
* We could penalize larger edge weights, in order to get a sparser or more regularized model.
* Weight momentum: the weights added to each edge are a weighted sum of the current step as described above (i.e., the $L * (Positive(e_{ij}) - Negative(e_{ij})$ from the current step) and the step previously taken.
* Batch learning: instead of using only one training example in each epoch, we can use larger batches instead (and only update the network's weights after passing through all the examples in the batch). This can speed up the learning by taking advantage of fast matrix-multiplication algorithms.

# Actual Code and Examples

Alright, so let's run some actual examples to see how an RBM performs for real. I initialize an RBM class

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

with 10 visible units and 2 hidden units:

	r = RBM(num_visible = 10, num_hidden = 2)

and train the network using the movie preferences above:

	data = np.array([[1,1,1,1,1,0,0,0,0,0],[0,0,1,0,0,1,1,1,1,0],[1,1,1,1,0,0,0,0,0,0],[0,0,1,0,0,0,1,1,1,0], [0,0,1,0,0,0,1,1,1,0],[1,1,1,1,1,0,0,0,0,0]])
	r.train(data, max_epochs = 1000)

	def train(self, data, max_epochs = 1000):
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
	    # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
	    # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
	    # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
	    pos_associations = np.dot(data.T, pos_hidden_probs)

	    # Reconstruct the visible units and sample again from the hidden units.
	    # (This is the "negative CD phase", aka the daydreaming phase.)
	    neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
	    neg_visible_probs = self._logistic(neg_visible_activations)
	    neg_visible_probs[:,0] = 1 # Fix the bias unit.
	    neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
	    neg_hidden_probs = self._logistic(neg_hidden_activations)
	    # Note that we're using the activation *probabilities* when computing associations, not the states 
	    # themselves. We could also use the states; see section 3 of Hinton's "A Practical Guide to Training 
	    # Restricted Boltzmann Machines" for more.
	    neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

	    # Update weights.
	    self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)

	    error = np.sum((data - neg_visible_probs) ** 2)
	    print "Epoch %s: error is %s\n" % (epoch, error)

Here are the weights learned by the network:

					  Bias Unit	   Hidden 1		Hidden 2
	Bias Unit	-0.02264486	-0.14238624	 0.39183875 
	LOTR 1		-2.58092036	 3.92514983	-1.62482845  
	LOTR 2		-2.42818272  3.94131185	-1.77020683  
	LOTR 3		 1.99969113  3.63940941	 2.85451444  
	HP				-2.46372953  3.94422824	-1.7396039  
	Twilight		-2.10073327  0.85324896	-2.71302185 
	Titanic		-0.19893787 -3.72986034	-0.49620514 
	Gladiator	 2.29375224 -4.01030233	 2.03001262 
	Beaut. Mind	 2.32403228 -4.01407343	 2.00364043 
	Slumdog		 2.32042494 -4.01066577	 2.00578411 
	Justin B.	-2.25949729 -3.68399181	-2.50880187

We see that all the SF&F units are connected to the first hidden unit with a positive weight, and all the Oscar winners are connected to the second hidden unit with a positive weight, so our RBM has correctly learned what we were hoping.

And if a new user comes in, who wants to watch all the Oscar winners except LOTR 3, we correctly surmise he likes Oscar winners in general, but not SF&F movies:

	new_user_preferences = np.array([[0,0,0,0,0,1,1,1,1,0]])
	new_user_hidden_states r.run(new_user_preferences)
	print new_user_hidden_states # outputs [0 1]