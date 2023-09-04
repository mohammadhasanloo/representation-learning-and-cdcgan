# NN-CA6-1.Variational-Auto-Encoder-2.Conditional-Deep-Convolutional-GAN

### 1.Variational Auto Encoder [Link](#part-1-variational-auto-encoder)

### 2.cGAN with BreastMNIST Dataset [Link](#part-2-conditional-deep-convolutional-gan)

# Part 1: Variational Auto Encoder

This report details the implementation of encoder-decoder networks based on the article titled "Empirical Comparison between Autoencoders and Traditional Dimensionality Reduction Methods" (link: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8791727)).

### 1. Dataset

- Dataset Used: -10CIFAR
- Student ID: Odd number
- Preprocessing: Max-min normalization to the [0, 1] range
- Data Reshape: One-dimensional arrays for compatibility with scikit-learn's dimensionality reduction algorithms

#### 1.1 PCA and Isomap

- Evaluation of KNN classifier accuracy after dimensionality reduction
- Random search function for parameter optimization (dimensions and neighbors)

##### Principal Component Analysis (PCA)

- Dimensionality reduction while preserving information
- Visualized top 10 principal components
- Highest accuracy achieved: 41.5%

##### Isomap

- Dimensionality reduction based on nearest neighbor distances
- Retains data structure and distribution
- Highest accuracy achieved: 27.10%

### 2. Encoder-Decoder Networks

- Manually adjusted settings
- Dense Autoencoder: Highest accuracy - 44.91%
- Convolutional Autoencoder: Regularized with Batch Normalization
- Dense models outperformed Convolutional models

### 3. Variational Autoencoders (VAE)

- Generative models combining deep learning and probabilistic modeling
- Transform data into a lower-dimensional latent space
- Reconstruction and posterior distribution learning
- Loss function combines reconstruction error and KL divergence

#### VAE Latent Space

- Dense VAE: Highest accuracy - 43.61%
- Regularized with Batch Normalization
- Final model with a latent space of 90

### 4. Latent Space Visualization

- Scatter plots of training data in the latent space
- Points tightly clustered, indicating successful VAE training

This report demonstrates the implementation and evaluation of encoder-decoder networks, providing insights into dimensionality reduction techniques and the effectiveness of Variational Autoencoders for feature extraction.

# Part 2: Conditional-Deep-Convolutional-GAN

In this part of the project, we implement the network structure by referencing the article titled "Conditional Generative Adversarial Nets" ([Link](https://arxiv.org/pdf/1411.1784.pdf)).

### Data Preprocessing and Augmentation

Before diving into the details of the conditional DCGAN, let's briefly discuss data preprocessing and augmentation techniques applied to the dataset:

- **RandomHorizontalFlip**: The input images are horizontally flipped with a 50% probability.
- **RandomVerticalFlip**: The input images are vertically flipped with a 50% probability.
- **RandomRotation**: Images are randomly rotated by a specified angle (15 degrees in our case).
- **ColorJitter**: Random alterations in brightness, contrast, saturation, and hue are applied to the input images.
- **RandomResizedCrop**: The initial image is randomly cropped to a size of 28x28 pixels with a random aspect ratio ranging from 0.8 to 1.
- **ToTensor**: Images are converted to tensor format.
- **Normalize**: Images are normalized with a mean of 0.5 and a standard deviation of 0.5 (for grayscale images).

### Dataset and Network Architecture

We utilized the **BreastMNIST** dataset for this part and implemented the conditional DCGAN architecture as described in the paper. The network architecture is designed to adhere to the paper's specifications, including:

- Batch size of 128.
- Stochastic Gradient Descent (SGD) optimizer.
- Weight initialization with a normal distribution centered at zero and a standard deviation of 0.02.
- LeakyReLU activation function with a slope of 0.2 for the discriminator.
- Adam optimizer for both generator and discriminator.
- Learning rate of 0.0002, as higher values were found to be suboptimal during experimentation.
- Momentum of 0.5 in the Adam optimizer to stabilize training.
- No data augmentation, only resizing to 64x64 and normalization to mean 0.5 and standard deviation 0.5.

### Conditional DCGAN Implementation

The implementation of the conditional DCGAN was divided into two parts. In the first part, we trained the network for 600 epochs without performing optimization to ensure convergence. In the second part, we trained the network while following the optimization procedure mentioned in the paper.

#### cGAN (Based on Keras)

First, we loaded the dataset into numpy arrays, applied one-hot encoding to the labels, and resized the images to 28x28 pixels with a single channel. Then, we defined the architecture of the cGAN with specific hyperparameters:

- Learning rate set to 0.0003 for both generator and discriminator.
- 1000 epochs of training.
- Both generator and discriminator losses converged to approximately 0.7.

#### Conditional DCGAN (Based on Paper)

The conditional DCGAN architecture was implemented according to the paper's specifications, with the following key modifications:

- Replaced all pooling layers with convolutional layers with strided convolutions.
- Introduced batch normalization in both the generator and discriminator.
- Removed fully connected layers from the architecture.
- Used ReLU activation for generator layers except the output layer, which used tanh.
- Employed LeakyReLU activation for all discriminator layers.
- The generator progressively increased the number of features until reaching the desired number of channels (1 for grayscale images).

### Training and Results

Training the network for 600 epochs allowed both the generator and discriminator losses to converge, indicating successful learning. However, the generated images showed limited quality due to the relatively low number of epochs. It's important to note that even the original BreastMNISt dataset contains images that are challenging to distinguish, making it difficult for the generator to produce highly realistic images.

As shown in the results, the network's accuracy improved during training, with training accuracy reaching 99%, validation accuracy around 68%, and test accuracy around 80%. These accuracy values suggest that the network might have encountered some degree of overfitting on the training data.

### Loss Functions in GAN

The loss functions for the discriminator and generator are critical in GANs. The discriminator's loss, typically based on Binary Cross-Entropy, measures how well it can distinguish between real and fake data. The generator's loss, also based on Binary Cross-Entropy, encourages the generator to produce data that the discriminator cannot easily differentiate from real data.

Throughout the training process, the loss functions for both the discriminator and generator should be balanced to ensure stable and effective training.
