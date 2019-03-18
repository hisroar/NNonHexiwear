# An example of human activity recognition using Tensorflow

## Notebook overview

Open [HAR_TF_Example.ipynb][./HAR_TF_Example.ipynb] for more details.

Run all in order to get a .pb file in the [output/][./output] directory, or just use the previously trained .pb files in the directory.

Copy the [output/] directory over to [HexiwearHARExample/] to be able to implement the model on Hexiwear.

A quick TL;DR of the notebook:

  - I used UCI's HAR dataset.
  - I used Tensorflow to train a DNN to recognize different human activities (about 70% success rate).
  - After training the DNN, it is saved as a .pb file, which will be translated into C code by uTensor.

Follow the set-up instructions below if you want to run the notebook.

## Getting started

### Tensorflow

To start, you must install Tensorflow. From what I've read online, it's probably best if you work in an Anaconda environment (also, this way you can open and run my .ipynb using Jupyter Notebook).

To install Anaconda, go to [their website][Anaconda] and follow the instructions.

Once Anaconda is installed (meaning you can run `conda`):

### If you have a dedicated GPU (e.g. NVIDIA or AMD, this will speed up training):

I followed [this guide][TF-GPU]. Skip to the section "The Award Winning New Approach". TL;DR below.

First, update Anaconda.
```sh
conda update conda
```

Run the following command anywhere:
```sh
conda create --name tf_gpu tensorflow-gpu 
```
This will create an environment called `tf_gpu` and install `tensorflow-gpu` on it. Use this environment whenever you try to run Tensorflow using your GPU. It shouldn't require any additional set-up to run TF on your GPU, it should really just work.

Next, activate the environment and install Jupyter Notebook (necessary, otherwise it will run Jupyter in a different environment without TF) and additional dependencies.
```sh
conda activate tf_gpu
conda install jupyter matplotlib scikit-learn
```

Now you should be able to open and run the notebook.
```sh
jupyter notebook
```

### Otherwise (no dedicated GPU):

First, update Anaconda.
```sh
conda update conda
```

Run the following command:
```sh
conda create --name tf tensorflow 
```

Next, activate the environment and install Jupyter Notebook (necessary, otherwise it will run Jupyter in a different environment without TF) and additional dependencies.
```sh
conda activate tf
conda install jupyter matplotlib scikit-learn
```

Now you should be able to open and run the notebook.
```sh
jupyter notebook
```

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Anaconda]: <https://www.anaconda.com/distribution/>
   [TF-GPU]: <https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc>
   [HAR_TF_Example.ipynb]: <https://github.com/hisroar/NNonHexiwear/blob/master/HAR_TF_Example/HAR_TF_Example.ipynb>