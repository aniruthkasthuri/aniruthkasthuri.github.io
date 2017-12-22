#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
import time
import matplotlib.pyplot as plt


# Some global variables.
PERCEPTRON_MAX_ITERATIONS = 2
PERCEPTRON_DIMENSION = 5

# A Python class to manage vector objects. Includes magnitude,
# addition, subtraction, dot product, scalar multiplication, etc.
# stolen from https://gist.github.com/mcleonard/5351452.
# PLEASE DO NOT MODIFY!

class Vector(object):
    """ 
    A Python class to manage vector objects. Includes magnitude,
    addition, subtraction, dot product, scalar multiplication, etc.
    stolen from https://gist.github.com/mcleonard/5351452.
    PLEASE DO NOT MODIFY!
    """

    def __init__(self, *args):
        """ Create a vector, example: v = Vector(1,2) """

        if len(args) == 0:
            self.values = (0, 0)
        else:
            self.values = args

    def dot(self, other):
        """ Returns the dot product (inner product) of self and other vector
        """

        return sum(a * b for (a, b) in zip(self, other))

    def norm(self):
        """ Returns the norm (length, magnitude) of the vector """

        return math.sqrt(sum(comp ** 2 for comp in self))

    def normalize(self):
        """ Returns a normalized unit vector """

        norm = self.norm()
        normed = tuple(comp / norm for comp in self)
        return Vector(*normed)

    def __iter__(self):
        return self.values.__iter__()

    def __mul__(self, other):
        """ Returns the dot product of self and other if multiplied
            by another Vector.  If multiplied by an int or float,
            multiplies each component by other.
        """

        if type(other) == type(self):
            return self.inner(other)
        elif type(other) == type(1) or type(other) == type(1.0):
            product = tuple(a * other for a in self)
            return Vector(*product)

    def __rmul__(self, other):
        """ Called if 4*self for instance """

        return self.__mul__(other)

    def __div__(self, other):
        if type(other) == type(1) or type(other) == type(1.0):
            divided = tuple(a / other for a in self)
            return Vector(*divided)

    def __add__(self, other):
        """ Returns the vector addition of self and other """

        added = tuple(a + b for (a, b) in zip(self, other))
        return Vector(*added)

    def __sub__(self, other):
        """ Returns the vector difference of self and other """

        subbed = tuple(a - b for (a, b) in zip(self, other))
        return Vector(*subbed)

    def __getitem__(self, key):
        return self.values[key]

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return str(self.values)




class PerceptronPlot:
    """ 
    A Python class to manage the perceptron graph. Includes a function
    to query the user for points, graph points, etc.
    PLEASE DO NOT MODIFY!
    """

    def set_dim(self, axes, dim):
        """ Set the max/min dimensions of the plot. """
        axes.set_xlim([-dim, dim])
        axes.set_ylim([-dim, dim])

    def ask_for_points(self):
        """ Query the user to enter points and return the result. """
        plt.pause(0.05)
        examples = []
        points = []

        # On click, record the coordinates and graph them.
        def callback(event):
            if event.xdata is not None and event.ydata is not None:
                if event.dblclick:
                    # A hack. On double click, delete the previous single click.
                    del examples[-1]
                    examples.append((Vector(event.xdata, event.ydata), 1))
                    (pt, ) = plt.plot(event.xdata, event.ydata, 'ro')
                    points.append(pt)
                else:
                    examples.append((Vector(event.xdata, event.ydata), -1))
                    (pt, ) = plt.plot(event.xdata, event.ydata, 'bo')
                    points.append(pt)

        self.fig.canvas.callbacks.connect('button_press_event', callback)

        while True:
            k = input("Press 'd' when you're done, or 'c' to clear: ")
            if k == 'd':
                if len(examples) == 0:
                    print('You need at least one point!')
                else:
                    break
            if k == 'c':
                # Remove all the points.
                examples = []
                for pt in points:
                    pt.remove()

        plt.close()
        return examples

    def setup_graph(self):

        # center the graph

        self.ax.spines['left'].set_position('center')
        self.ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes

        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only

        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')

        self.ax.set_aspect('equal')

        axes = plt.gca()
        self.axes = axes
        self.set_dim(self.axes, self.dim / 2.)

    def __init__(self, dimension):
        self.dim = dimension

        (fig, ax) = plt.subplots()
        self.ax = ax
        self.fig = fig
        plt.ion()
        self.setup_graph()

    def plot_decision_boundary(self, theta, theta0=0):

        # this is to make sure we're scaling the decision boundary to fit the plot

        scale = min(abs(self.dim / theta[1]), abs(self.dim / theta[0]))

        # some hacky scaling stuff to print lines better

        REALLY_LARGE_DIM = 100000000
        self.set_dim(self.axes, REALLY_LARGE_DIM)
        LARGE_DIM = 100

        # when we print theta, define its length and normalize theta to that length

        theta_length = min(2, self.dim / 4)
        scaled_theta = theta_length * Vector(*theta).normalize()

        y_intercept = -theta0 / theta[1]

        # print theta as an arrow

        self.ax.arrow(
            0,
            y_intercept,
            scaled_theta[0],
            scaled_theta[1],
            head_width=0.2,
            head_length=0.2,
            fc='k',
            ec='k',
            )

        # print the decision boundary as a line

        self.ax.plot((LARGE_DIM * theta[1], -LARGE_DIM * theta[1]),
                     (-LARGE_DIM * theta[0] + y_intercept, LARGE_DIM
                     * theta[0] + y_intercept), '#3498db', lw=3)

        # fix the dimensions

        self.set_dim(self.axes, self.dim / 2.)

    def plot_examples(self, examples):
        for (x, y) in examples:
            plt.plot(x[0], x[1], ('ro' if y == 1 else 'bo'))

    def plot(self, examples, theta, theta0, circle_point=None):
        plt.cla()
        self.setup_graph()

        for (x, y) in examples:
            plt.plot(x[0], x[1], ('ro' if y == 1 else 'bo'))

        if circle_point is not None:
            plt.plot(
                circle_point[0],
                circle_point[1],
                'go',
                mfc='none',
                ms=50,
                mew=5,
                )

        self.plot_decision_boundary(theta, theta0)
        plt.pause(0.05)



class Perceptron:
    """ 
    A Python class to run Perceptron.
    """

    def __init__(
        self,
        training_examples,
        max_iterations=PERCEPTRON_MAX_ITERATIONS,
        ):

        self.num_iterations = 0
        self.wait_for_keypress = True
        self.training_examples = training_examples
        # the dimension, or the number of features, of the training examples
        self.dim = len(self.training_examples[0])
        self.max_iterations = max_iterations

    def train(self):
        # initialize theta and theta0 to random values
        self.theta = Vector(random.random(), random.random())
        self.theta0 = 0

        self.plot_obj = None

        # If it's two-dimensional, we can graph it. 
        if self.dim == 2:
            # Find the right dimensions of the graph.
            plot_dim = max(PERCEPTRON_DIMENSION/2., 1.2 * max(max(x) for (x, y) in self.training_examples))
            self.plot_obj = PerceptronPlot(plot_dim * 2)

        # Plot the initial graph.
        self.plot()

        # Prompt user to start.
        self.plot_and_wait()

        # Train the model!
        self.run_perceptron()

        # Check if we converged or not.
        if self.num_iterations < self.max_iterations:
            print('Converged! :)')
            print('theta = ', self.theta)
            print('theta0 = ', self.theta0)
        else:
            print('Did not converge after', self.max_iterations,
                   'iterations. :(')

        self.plot()
        plt.show()

        while input("Press 'q' to quit: ") != 'q':
            pass

        plt.close()

    def plot_and_wait(self, p=None):
        """ Plot the points, and wait for the user to continue. """
        self.plot(circle_point=p)
        if not self.wait_for_keypress:
            time.sleep(0.02)
            return

        while True:
            k = input("Press 's' to step, or 'c' to continue: ")
            if k == 's':
                break
            if k == 'c':
                self.wait_for_keypress = False
                break

    def run_perceptron(self):
        """ Train perceptron! Your job is to figure out this function. """
        self.num_iterations = 0
        while True:
            mistakes = False
            for x, y in self.training_examples: 
                # FILL IN THIS CODE
                h = None
                if h != y:
                    mistakes = True
                    # FILL IN THIS CODE
                    self.plot_and_wait(x)

            # Keep this code.
            if not mistakes:
                break

            self.num_iterations = self.num_iterations + 1
            if self.num_iterations == self.max_iterations:
                break

    def plot(self, circle_point=None):
        """ Plot the points. """
        if self.plot_obj is not None:
            self.plot_obj.plot(self.training_examples, self.theta,
                               self.theta0, circle_point=circle_point)


if __name__ == "__main__":
    p = PerceptronPlot(PERCEPTRON_DIMENSION)
    examples = p.ask_for_points()
    p = Perceptron(examples)
    p.train()

