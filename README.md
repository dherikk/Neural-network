# Neural network

This is a neural network, which can be used as a universal function. If there exists a function $$f : S \rightarrow \mathbb{R}$$ where we let $$S \subseteq \mathbb{R}^{n}$$ where $$\delta : \mathbb{R} \rightarrow \mathbb{R}$$ and $$\sigma > 0$$ we can write $$g(x) = \sum_{i=1}^{d} c_k \delta (a_k + b_{j}^{\top}x)$$ such that $$|g(x) - f(x)| < \sigma$$

This essentially means for any given continous function, there exists a neural network that will approximate the given function.

This repository contains an implementation of a feedforward neural network.
I have used a training set to train this neural network to output a classification based on the input. The training data is from the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database. To simplify my classifications, I have used the datasets of 9's and 3's to attempt to correctly classify these using the datasets as training data. The classification function is tried applied to *k* fold batches of the dataset, in addition to a test dataset.

## 