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