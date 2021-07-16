# A quick introduction to the basics of deep learning

This repository contains materials for a quick introductory course on the basics of deep learning for the participants of the [data science for social good summer program](https://datascience.stanford.edu/programs/data-science-social-good-summer-program) of Stanford Data Science.

Each topic is covered in a separate [Jupyter notebook](https://jupyter.org); each notebook contains a brief theoretical introduction to its topic as well as a practical exercise.


## 1. Running the notebooks

For a general introduction to the Jupyter environment, I recommend [this tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/).

You can either run the Jupyter notebooks locally on your personal computer (see below for the installation instructions) or remotely with [Jupyter Binder](https://mybinder.org) using the following link: https://mybinder.org/v2/gh/athms/dssg-dl-tutorial/HEAD

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/athms/dssg-dl-tutorial/HEAD)


## 2. Local installation

**1. Clone and switch to this repository:**

```bash
git clone https://github.com/athms/dssg-dl-tutorial.git
cd dssg-dl-tutorial
```

**2. Install all dependencies** listed in [`requirements.txt`](requirements.txt). 

I recommend setting up a new Python environment (e.g., with the [miniconda installer](https://docs.conda.io/en/latest/miniconda.html)). 

You can create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using the following command:

```bash
conda create --name dssg-dl-tutorial python=3.8
```

This will create a Python 3.8 environment with the name `dssg-dl-tutorial` .

You can activate the new environment as follows:

```bash
conda activate dssg-dl-tutorial
```

and then install all required dependencies with: 

```bash
pip3 install -r requirements.txt
```

**3. Start the Jupyter notebook server:**

```bash
jupyter notebook
```
