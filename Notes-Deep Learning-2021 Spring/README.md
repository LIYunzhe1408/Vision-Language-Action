## Intuition of deep learning
Google wants to have a multilingual translation model to take sentences in random language as input and output desired language to replace separate models to deal with corresponding language.
Take a step further, this multilingual model can process mix of language.(i.e. english to mix of 40% Russian and 60% Belarusian). But anecdote may happen the output might not be both Russian and Belarusian when the mix ratio is 50% 50%.
Why this anecdote happen and how they did it? The separate one just translate model while the multilingual one understand the thoughts or semantic under the sentence and translate this representation into the desired output.
The representation is the key in deep learning, that's what deep learning learns from data.

### What is machine learning
* Before machine learning, rules are defined by hand and inputs cannot be efficiently mapped to desired outputs, especially in special cases & exceptions.

Regarding machine learning, it's a program, instead of rules defined by hand, that learns new stuff from data and examples and relationship between input and output. 
* Say, take a puppy image as input, processed by this program that has learnt representations from vast data, then output the object label.
* You can certainly define rules by hand like 2 ears with one nose, but what if the rules are too complex or what if we don't exactly know the rules. And there should be too many exceptions and special cases.

Another example is that, how to separate green cross and red circles? To parameterize a function that gives the correct answer(draw a line). The function should be $x_1\theta_{1}+x_2\theta_{2}+\theta_{3}<0$. $\theta_{1}, \theta_{2}, \theta_{3}$ are what programs or human learn. Then the parameterized function can be generalized as $f(x, \theta)=y$ or $f_{\theta}(x)=y$. The input with the learnt parameters gets the output. This $f_{\theta}(x)$ can be almost any expression of $x$ and $\theta$.

### Shallow learning to deep learning
For the puppy example, the 'drawing a line' function is not workable. How to parameterize?
* A compromise solution is not hand-program the rules but hand-program the features. Say, use fixed function for extraction features from $x$. Regard $f_\theta(x)=y$ as $\phi(x)^T\theta\le{0}$, the $\phi(x)$ is hand-programed.
* Leaning on top of the features extracted from $\phi(x)$ can be simple but coming up with good features is very hard.

So, for deep learning, we learn parameters not only for $\phi(x)^T\theta\le{0}$ but also for $\phi(x)$ as $\phi_\theta(x)$.
* Why it is called 'deep'? It's the machine learning with multiple layers of learnt representations. Each layer represents a **simple parameterized** transformation for different level features.
* Not leveraging the hand-program feature extractor contributes why it is called end-to-end learning.

### What makes deep leaning work?
* Big models with many layers
* Large datasets with many examples
* Enough compute to handle all this

Underlying themes include the consideration of **model capacity**, **learning+inductive bias**, and **ability of algo to scale** with more data and larger capacity.

The reason why it's called neural network is it adopted the concept of transmitting signals between neurons as activation function to determine whether summing up signals and fire to downstream units.


## Machine Learning Basics
### Learning Problems
There are several types of problems in machine learning: supervised learning, unsupervised learning, and reinforcement learning.
* Supervised learning is to learn a function $f_\theta(x)$ to make it approximate to the **given label** $y$. Predict $y$ based on the $x$. Questions to answer:
  * How to represent $f_\theta(x)$: linear function or polynomial function
  * How to measure difference between $f_\theta(x)$ and label $y$: loss function
  * How to find the best setting of $\theta$: Optimization
* Unsupervised learning learns underlying representations, leading to generative modeling. They are trained without provided labels. The images that were provided to the model came from some underlying distribution. They just reconstruct the set and output resembling the one in the training set.
* Reinforcement learning is to learn a function $f_\theta(x)$ as action to maximize the reward, which is similar to supervised learning. Supervised learning is to get $f_\theta(x)$ to match $y_i$, while RL is to get $f_\theta(x)$ to maximize reward(can be anything).
  * Action: bark; observation: sight, smell; reward: food
  * Action: motor current or torque; observation: camera images; rewards: running speed
  * Action: what to purchase; observation: inventory levels; reward: profit

### Predicting probabilities
When predict the correct number of handwritten digits, human tends to give probabilities of possible answers. In this way, it also makes more sense for machine to predict possible label probabilities rather than yes-or-no for particular digit.

Guided by this intuition, instead of learning $f_\theta(x)\approx{y}$, we learn $p_\theta(y|x)$ which is a probability distribution.
* Conditional probabilities:
  * $x$ is random variable representing the input: we dont know what $x$ we're going to get, so it's random
  * $y$ is random variable representing the output
  * $p(x,y)=p(x)p(y|x)$
  * Two ways to get $p(y|x)$: discriminative way to directly learn $p(y|x)$ and generative way to generate $x$s to get $p(x,y)$ then based on chain rule to get $p(y|x)$

How to represent $p(y|x)$?
* $p(y=dog|x)=x^T\theta_{dog}$
* $p(y=cat|x)=x^T\theta_{cat}$
* $\theta=\{\theta_{dog}, \theta_{cat}\}$
* These two $p$ should sum to 1.0, which is done by `softmax`.
* `softmax` can be any (ideally one to one & onto) function that takes inputs and outputs probabilities that are positive and sum to 1.
  * Make number positive: $z^2$, $|z|$, $max(0, z)$, $\exp(z)$
  * Make a bunch of numbers sum to one:$\frac{z_1}{z_1+z_2}$, $\frac{z_1}{\sum_{i=1}^{n}z_i}$

In general, `softmax` works as follows:
* $N$ possible labels
* $p(y|x)$ is vector with $N$ elements
* $f_\theta(x)$ is vector-valued function with $N$ outputs
* $p(y=i|x)=softmax(f_\theta(x))[i]=\frac{\exp{(f_{\theta,i}(x))}}{\sum_{j=1}^{N}\exp(f_{\theta,j}(x))}$

Why is it called `softmax`?
* The larger the coefficient, the sharper the outputs.
* When it gets 100, it will have a very sharp\rapid change in the middle, like a step function.
* It's just like `max` function. You are just assigning a high probability to the label with the largest $\theta^Tx$, others are zero.


### How to select $\theta$ Lect-2 Pic 16