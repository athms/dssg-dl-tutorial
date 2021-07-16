#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from seaborn import despine
from sklearn.metrics import confusion_matrix

from .perceptron import cross_entropy_loss, sigmoid, step


def plot_iris_dataset(data, classes=np.arange(3), features=['sepal', 'petal']):
    """Visualizes the iris dataset, by plotting all insatnces
    according to their sepal and petal width and length

    Args:
        data ([dict]): data dictionary as obtained
            from sklearn.datasets.load_iris()
        classes ([array of ints or strings], optional): array of class labels 
            (either ints or strings). Defaults to np.arange(3).
        features (list, optional): names of data features.
            Defaults to ['sepal', 'petal'].

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    assert np.all([f in ['sepal', 'petal'] for f in features]), 'invalid features; need to be sepal or petal'
    if not np.all([c in np.arange(3) for c in classes]) and not np.all([c in ['setosa', 'versicolor', 'virginica'] for c in classes]):
        raise ValueError('classes need to be in [0,1,2] or [setosa, versicolor, virginica]')
    # make figure
    fig, axs = plt.subplots(1,2,figsize=(12,6), dpi=100)
    # iterate iris types
    for i, iris_type in enumerate(data['target_names']):
        if i in classes or iris_type in classes:
            # idx iris type
            idx = data['target']==i
            if 'sepal' in features:
                # plot sepal length / width
                axs[0].scatter(data['data'][idx,0], data['data'][idx,1],
                            color='C{}'.format(i),
                            alpha=0.75,
                            label=iris_type.capitalize())
            if 'petal' in features:
                # plot petal length / width
                axs[1].scatter(data['data'][idx,2], data['data'][idx,3],
                            color='C{}'.format(i),
                            alpha=0.75,
                            label=iris_type.capitalize())
    # add labels
    axs[0].set_title('Sepal')  
    axs[0].set_xlabel(data['feature_names'][0].capitalize())    
    axs[0].set_ylabel(data['feature_names'][1].capitalize()) 
    if 'sepal' in features:
        axs[0].legend(loc='upper left')
    axs[0].set_xlim(0,8)
    axs[0].set_ylim(0,5)
    axs[1].set_title('Petal')  
    axs[1].set_xlabel(data['feature_names'][2].capitalize())    
    axs[1].set_ylabel(data['feature_names'][3].capitalize()) 
    axs[1].set_xlim(0,8)
    axs[1].set_ylim(0,5)
    if 'petal' in features:
        axs[1].legend(loc='upper left')
    # save
    for ax in axs:
        despine(ax=ax)
    fig.tight_layout()
    
    return fig, axs


def plot_perceptron_loss(X, y, w1grid=np.linspace(-10, 30, 100),
    w2grid=np.linspace(-10, 30, 100), b=-19, dimensions='2d'):
    """Visualizes the cross entropy loss of 
    our perceptron implementation for two weights (excluding the bias!)

    Args:
        X ([ndarray]): the data (instances x features)
        w1grid, w2grid ([array], optional): plotted weight parameter values 
        b (float, optional): bias value
        dimensions ([string], optional): whether to plot the loss
            as a color ("2d") or as a thrid dimension ("3d"). 
            Defaults to 2d.

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    # initiate our loss
    xe_loss = cross_entropy_loss()

    # define the grid of w-values for which we want to compute the loss:
    ww1, ww2 = np.meshgrid(w1grid, w2grid) # grid indices for w1 and w2 values

    # compute the loss for each point on the grid
    zz = np.zeros(ww1.shape)
    for i in range(zz.shape[0]):
        for j in range(zz.shape[1]):
            w = np.array([ww1[i,j], ww2[i,j]])
            y_pred = sigmoid(X.dot(w)+b)
            # we average the loss over all data instances
            zz[i, j] += np.mean(xe_loss.loss(y, y_pred))

    if dimensions == '2d':
        # setup figure
        fig, ax = plt.subplots(1, 1, figsize=(8,6), dpi=100)
        # plot contour
        cs = ax.contourf(ww1, ww2, zz, 100, cmap=cm.viridis)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
        # label axes
        cbar.set_label('Loss')
        ax.set_xlabel(r"$w_1$")
        ax.set_ylabel(r"$w_2$")
        # save figure
        fig.tight_layout()

    elif dimensions == '3d':
        # setup figure
        fig = plt.figure(figsize=(8,6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        # plot surface
        ax.plot_surface(ww1, ww2, zz, alpha=0.75, cmap=cm.viridis, 
                        linewidth=0, antialiased=False, zorder=-99)
        # add labels
        ax.set_xlabel(r"$w_1$")
        ax.set_ylabel(r"$w_2$")
        ax.set_zlabel('Loss', labelpad=7) 
        ax.tick_params(labelsize=10) # increase labelsize
        fig.tight_layout()

    else:
        raise ValueError('dimensions needs to be 2d or 3d')

    return fig, ax


def plot_activation_functions():
    """Visualize the step and sigmoid
    activation functions.

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    # setup figure
    fig, axs = plt.subplots(1,2,figsize=(12,6), dpi=100)

    # set input range
    x = np.linspace(-5,5,100)

    # Step activation
    axs[0].set_title('Original perceptron uses:\nStep activation')
    axs[0].plot(x, step(x), lw=5)
    axs[0].set_ylabel(r'$\phi(x)$')
    axs[0].set_xlabel(r'$x$')

    # Sigmoid activation
    axs[1].set_title('For simplicity, we use:\nSigmoid activation')
    axs[1].plot(x, sigmoid(x), lw=5)
    axs[1].set_ylabel(r'$\phi(x)$')
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylim(0, 1)

    # despine
    for ax in axs:
        despine(ax=ax)
    fig.tight_layout()

    return fig, axs


def plot_cross_entropy_loss():
    """Visualize the two components
    of the binary cross entropy loss.

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    # setup figure
    fig, axs = plt.subplots(1,2,figsize=(12,6),dpi=100)

    # predicted probability that y = 1
    p = np.linspace(1e-2,1-1e-2,100)

    # plot loss if y = 1
    axs[0].plot(p, -np.log(p), lw=5)
    axs[0].set_title('If y = 1')
    axs[0].set_ylabel(r'$-log(p)$')
    axs[0].set_xlabel(r'$p(y=1)$')
    despine(ax=axs[0])

    # plot loss if y = 0
    axs[1].plot(p, -np.log(1-p), lw=5, color='red')
    axs[1].set_title('If y = 0')
    axs[1].set_ylabel(r'$-log(1-p)$')
    axs[1].set_xlabel(r'$p(y=1)$')
    despine(ax=axs[1])

    return fig, axs


def plot_gradient_descent_path(perceptron, X, y):
    """Plot the gradient descent path of
    a trained perceptron.

    Args:
        perceptron ([class]): trained perceptron. Expects
             instance of perceptron.Perceptron.
        X ([ndarray]): the data (instaces x features)
        y ([array]): labels for each data instance

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    # extract the weights for each gradient step
    training_w = np.array(perceptron.training_w)
    b = float(perceptron.b) # also extract bias
    
    # define steps will be plotted
    n_steps = len(training_w)
    suggested_steps = np.array([0, 1, 2, 10, 50, 5000])
    steps = np.array([s for s in suggested_steps if s < n_steps])
    steps = np.append(steps, -1).astype(np.int)

    # compute the values of the loss function for a grid of w-values,
    # given our learned bias term
    w1_vals = np.linspace(np.min([np.min(training_w[:,0]), -10]),
                          np.max([np.max(training_w[:,0]), 30]), 100)
    w2_vals = np.linspace(np.min([np.min(training_w[:,1]), -10]),
                          np.max([np.max(training_w[:,1]), 30]), 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    grid_r, grid_c = W1.shape
    ZZ = np.zeros((grid_r, grid_c))
    for i in range(grid_r):
        for j in range(grid_c):
            w = np.array([W1[i,j], W2[i,j]])
            y_pred = perceptron.activation(X.dot(w)+b)
            ZZ[i, j] += np.nan_to_num(np.mean(perceptron.loss(y, y_pred)))

    # plot the loss function and gradient descent steps
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    
    # plot contour
    cs = ax.contourf(W1, W2, ZZ, 50, vmax=10, cmap=cm.viridis)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
    
    # plot trajectory
    ax.scatter(training_w[steps,0], training_w[steps,1], color='white', s=100)
    
    # mark start point
    ax.scatter(training_w[0,0], training_w[0,1], color='black', s=100, zorder=99)
    ax.plot(training_w[steps,0], training_w[steps,1], color='white', lw=1)
    
    # add line for final weights
    ax.axvline(training_w[-1,0], color='red', lw=1, ls='--')
    ax.axhline(training_w[-1,1], color='red', lw=1, ls='--')
    
    # label axes
    cbar.set_label('Loss')
    ax.set_title('Final loss: {}'.format(perceptron.training_loss[-1]))
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    
    return fig, ax


def plot_perceptron_decision_boundary(perceptron, X, y, x1label=None, x2label=None,
    y_labels=None, x1grid=np.linspace(0, 6, 50), x2grid=np.linspace(0, 2, 50)):
    """Plot the learned decision boundary of our
    trained perceptron. 

    Args:
        perceptron ([class]): trained perceptron. Expects
             instance of perceptron.Perceptron.
        X ([ndarray]): the data (instaces x features)
        y ([array]): labels for each data instance
        x1label, x2label ([string], optional): Labels of x- and y-axis
            (i.e., data features). Defaults to x1 and x2
        y_labels ([list], optional): Names of classes in y
        x1grid, x2grid ([array], optional): values of x1 and x2 to plot

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    # set default labels
    if x1label is None:
        x1label = 'x1'
    if x2label is None:
        x2label = 'x1'
    if y_labels is None:
        y_labels = np.array(['class-{}'.format(i)
            for i in range(np.unique(y).size)])
    
    # define a grid of x1 and x2 values for 
    # which we want to predict the probability
    # that each data point belongs to class 1
    # create all of the rows and columns of the grid
    xx1, xx2 = np.meshgrid(x1grid, x2grid)
    
    # flatten each grid to a vector
    x1, x2 = xx1.flatten(), xx2.flatten()
    x1, x2 = x1.reshape((-1, 1)), x2.reshape((-1, 1))
    
    # horizontal stack vectors to create x1, x2 input for the model
    grid = np.hstack((x1, x2))
    
    # predict probability that each point 
    # of the grid belongs to class 1
    zz = perceptron.predict(grid).reshape(xx1.shape)

    # setup figure
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    
    # plot predicted probabilities
    cs = ax.contourf(xx1, xx2, zz)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
    cbar.set_label('P(is {})'.format(y_labels[1]))
    
    # add scatter markers for instances of each iris type
    for cl in np.unique(y):
        idx = y==cl
        ax.scatter(x=X[idx,0],
                y=X[idx,1],
                color='C{}'.format(cl),
                label=y_labels[cl],
                alpha=1)
   
    # add labels
    ax.set_xlabel(x1label)
    ax.set_ylabel(x2label)
    ax.legend()
    despine(ax=ax)
    # save figure
    fig.tight_layout()

    return fig, ax


def generate_not_linearly_separable_clusters(n_samples=50):
    """Helper to generate clusters that are not 
    linearly separable in 2D.

    Args:
        n_samples (int, optional): Number of samples per 
            cluster. Defaults to 50.

    Returns:
        [ndarray]: x1 and x2 values for each instance
        [array]: label of each instance (0 or 1)
    """
    # generate random data
    X1 = []
    X2 = []
    # iterate samples
    for sample in range(n_samples):
        
        # class 1
        if np.random.uniform(0,1,1) > 0.5:
            X1.append(np.array([np.random.normal(-1,0.2,1),
                      np.random.normal(1,0.2,1)]).reshape(1,-1))
        else:
            X1.append(np.array([np.random.normal(1,0.2,1),
                      np.random.normal(-1,0.2,1)]).reshape(1,-1))
        
        # class 2  
        if np.random.uniform(0,1,1) > 0.5:
            X2.append(np.array([np.random.normal(-1,0.2,1),
                      np.random.normal(-1,0.2,1)]).reshape(1,-1))
        else:
            X2.append(np.array([np.random.normal(1,0.2,1),
                      np.random.normal(1,0.2,1)]).reshape(1,-1))
    
    # collect
    X = np.concatenate([np.concatenate(X1), np.concatenate(X2)])
    y = np.zeros((X.shape[0],1))
    y[n_samples:] = 1
    y = y.astype(np.int)
    
    return X, y


def plot_not_linearly_separable_clusters(X, y):
    """Plot clusters generated by 
     plotting.generate_not_linearly_separable_clusters.

    Args:
        X ([ndarray]): the data (instaces x features)
        y ([array]): labels for each data instance

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(X[(y==0).ravel(),0], X[(y==0).ravel(),1], c='b')
    ax.scatter(X[(y==1).ravel(),0], X[(y==1).ravel(),1], c='r')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    despine(ax=ax)
    fig.tight_layout()

    return fig, ax


def plot_XOR_solution(X, y):
    """Plot XOR solution that uses three perceptrons
    for clusters generated by plotting.generate_not_linearly_separable_clusters

    Args:
        X ([ndarray]): the data (instaces x features)
        y ([array]): labels for each data instance

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """

    # setup figure
    fig, axs = plt.subplots(1,3,figsize=(18,6))

    # Perceptron 1
    axs[0].set_title('Perceptron 1:\n'+'y=1, if '+r'$x_1<0$'+' and '+r'$x_2<0$')
    idx1 = np.logical_and(X[:,0]<0, X[:,1]<0)
    axs[0].scatter(X[idx1,0], X[idx1,1], color='red')
    axs[0].scatter(X[~idx1,0], X[~idx1,1], color='blue')

    # Perceptron 2
    axs[1].set_title('Perceptron 2:\n'+'y=1, if '+r'$x_1>0$'+' and '+r'$x_2>0$')
    idx2 = np.logical_and(X[:,0]>0, X[:,1]>0)
    axs[1].scatter(X[idx2,0], X[idx2,1], color='red')
    axs[1].scatter(X[~idx2,0], X[~idx2,1], color='blue')

    # Perceptron 3
    axs[2].set_title('Perceptron 3:\n'+'y=1, if Perceptr. 1 or Perceptr. 2')
    idx3 = np.logical_or(idx1, idx2)
    axs[2].scatter(X[idx3,0], X[idx3,1], color='red')
    axs[2].scatter(X[~idx3,0], X[~idx3,1], color='blue')

    # label axes
    for ax in axs:
        despine(ax=ax)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
    fig.tight_layout()

    return fig, axs


def plot_XOR_decision_boundaries(p1, p2, p3, X, y1, y2, y3,
    x1grid=np.linspace(-2,2,100), x2grid=np.linspace(-2,2,100)):
    """Plot decision boundaries of three trained perceptrons
    to solve XOR task; as generated by plotting.generate_not_linearly_separable_clusters.

    Args:
        p1, p2, p3 ([class]): trained perceptron. Expects
             instance of perceptron.Perceptron.
        X ([ndarray]): the data (instaces x features)
        y1, y2, y3 ([type]): labels for each data instance
        x1grid, x2grid ([type], optional): values of x1 and x2 to plot.
            Defaults to np.linspace(-2,2,100).

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    # setup a meshgrid (each x1 and x2 coordinate 
    # for which we want to predict a probability)
    xx1, xx2 = np.meshgrid(x1grid,x2grid)

    # predict with Perceptron 1 & 2
    zz1 = p1.predict(np.c_[xx1.ravel(), xx2.ravel()])
    zz2 = p2.predict(np.c_[xx1.ravel(), xx2.ravel()])

    # predict with Perceptron 3, based of prediction of perceptron 1 & 2
    zz3 = p3.predict(np.c_[zz1.ravel(), zz2.ravel()])

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(18,7))
    for i, (y, zz) in enumerate(zip([y1, y2, y3], [zz1, zz2, zz3])):
        ax = axs[i]
        cs = ax.contourf(xx1, xx2, zz.reshape(xx1.shape))
        cbar = fig.colorbar(cs, ax=ax, shrink=0.9, orientation="horizontal")
        cbar.set_label('P(is red)')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        for j in range(y.shape[0]):
            if y[j] == 0:
                marker = 'bo'
            else:
                marker = 'ro'
            ax.plot(X[j][0], X[j][1], marker)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_title('Decision boundary:\nPerceptron {}'.format(i+1))
    fig.tight_layout(w_pad=4)

    return fig, axs


def plot_confusion_matrix(y_true, y_pred, y_labels=None):
    """Plot a simply confusion matrix

    Args:
        y_true ([array]): True data labels (int)
        y_pred ([array]): Predicted data labels (int)
        y_labels ([array], optional): Labels of individual classes.
            Defaults to values of y_true.

    Returns:
        [matplotlib figure]
        [matplotlib axis]
    """
    if y_labels is None:
        y_labels = np.sort(np.unique(y_true))
    # make figure
    fig, ax = plt.subplots(dpi=150)
    cm = ax.imshow(confusion_matrix(y_true.ravel(), y_pred, normalize='true'))
    ax.set_xticks(np.arange(10)); ax.set_xticklabels(y_labels, rotation=90, fontsize=10);
    ax.set_yticks(np.arange(10)); ax.set_yticklabels(y_labels, fontsize=10);
    ax.set_title('Accuracy: {:.2f}%'.format(np.mean(y_pred==y_true.ravel()) * 100), fontsize=10)
    cbar = fig.colorbar(cm)
    cbar.set_label('Frequency (%)', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig, ax