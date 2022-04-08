# RL_GANs

In this repository, you will find many notebooks that can assist you in understanding how to train neural networks with `PyTorch`. 
You will find an example of Generative Adversarial Networks (GANs). GANs show how creative AI can be, from image-to-image translation to music generation and other exciting areas. The second example explores the possibilities we get with tools like `Wanda`. This tool has various features, such as experiment tracking, hyperparameters tracking, and hyperparameter tuning with sweeps.
The third example shows you the principle of data augmentation, why it is an essential tool for many machine learning algorithms, and how to augment our data set using classical methods and GANs.
In the reinforcement learning (RL) paradigm, an agent aims to learn behavior from interactions with an initially unknown environment. In our final notebook, you will comprehend how to train a Reinforcement Learning strategy in an OpenAI Gym environment. You will be able to follow the training performance using `WandB`. Finally, you will load the trained model and evaluate it in the environment. 
In this Readme, a small introduction shows you how you can train your neural network and run this notebook remotely on a server.  

## Contents:



## Prerequisites

1. The simulation experiments using mujoco-Py require a working install of [MuJoCo 2.1.0](https://github.com/deepmind/mujoco/releases). This version of Mujoco is an open sourece and requires no license.
2. We use conda environments for installs (tested on conda 4.* - 4.*), please refer to [Anaconda](https://docs.anaconda.com/anaconda/install/) for instructions.


## Installation

1. Clone this repository

```bash
git clone https://github.com/anyboby/Constrained-Model-Based-Policy-Optimization.git
```

2. Create a conda environment using the rl_gans yml-file

```bash
cd RL_GANs/
conda env create -f rl_gans.yml
conda activate rl_gans
```

3.Add the new kernel to the ipython kernels

```bash
ipython kernel install --user --name='rl_gans'
```

4. test your `mujoco-py` install using `python`

```python
import gym
env = gym.make('FetchPush-v1')
```

To deactivate an active environment, use `conda deactivate`

## Running your Notebook Remotly:

1. connect with your remote server:  

```bash
ssh user_name@server_ip
```

2. **Only In your first connection**, you may be prompted to enter an Access Token as typical to most Jupyter notebooks.  Normally, I'd copy-paste it from my terminal, but to make things easier for you, you can set-up your notebook password  

```bash
jupyter notebook password
```

3. Run Jupyter Notebook from a remote machine without browser

```bash
jupyter notebook --no-browser --port=XXXX
```

where

* `--port=XXXX`: this sets the port for starting your notebook where the default is `8888`. When it's occupied, it finds the next available port.

4. Forward port XXXX of the remote machine to YYYY of your device and listen to it  

```bash
ssh -N -f -L localhost:YYYY:localhost:XXXX user_name@server_ip
```

where

* `--port=XXXX` you need to use the same which you defined in remote machine 8889
* `--port=YYYY` 8889

5. Fire-up Jupyter Notebook:  start your browser and type the following in your address bar:
   `localhost:YYYY`

To see more information about the different arguments, you can see this [link](https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/)

