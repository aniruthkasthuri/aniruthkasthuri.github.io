#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
import time
import matplotlib.pyplot as plt


# Some global variables.
PLOT_DIMENSION = 5

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

	def __eq__(self, other):
		"""Override the default Equals behavior"""
		if isinstance(other, self.__class__):
			return self.values == other.values
		return False



class KMeansPlot:
	""" 
	A Python class to manage the KMeans graph. Includes a function
	to query the user for points, graph points, etc.
	PLEASE DO NOT MODIFY!
	"""
	def __init__(self, dimension):
		self.dim = dimension

		(fig, ax) = plt.subplots()
		self.ax = ax
		self.fig = fig
		plt.ion()
		self.setup_graph()


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
				examples.append(Vector(event.xdata, event.ydata))
				(pt, ) = plt.plot(event.xdata, event.ydata, 'ko')
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

   
	def plot(self, points, centers):
		colors = ['b', 'g', 'r', 'c', 'm', 'y']
		
		plt.cla()
		self.setup_graph()

		for x, c in points:
			if c == None:
				plt.plot(x[0], x[1], 'ko')
			else:
				plt.plot(x[0], x[1], colors[c] + 'o')

		for i in range(len(centers)):
			plt.plot(centers[i][0], centers[i][1], colors[i] + '+', mew=5, ms=20)

		plt.pause(0.05)


class KMeans2D:
	""" 
	A Python class to run KMeans.
	"""

	def __init__(
		self,
		k,
		points
		):

		if len(points[0]) != 2:
			raise Exception("k must equal 2")
		# the dimension, or the number of features, of the training examples
		self.kmeans = KMeans(k, points, self.plot_and_wait)
		self.wait_for_keypress = True

	def train(self):
		# initialize theta and theta0 to random values
		plot_dim = PLOT_DIMENSION/2.
		self.plot_obj = KMeansPlot(plot_dim * 2)

		# Plot the initial graph.
		self.plot()

		# Prompt user to start.
		self.plot_and_wait()

		# Train the model!
		self.kmeans.run_kmeans()

		self.plot()
		plt.show()

		# We're done, print the final message. 
		print("Converged in " + str(self.kmeans.num_iterations) + " iterations!")
		print("Loss is " + str(self.kmeans.loss()) + " .")
		print("Centers: ")
		for c in self.kmeans.centers:
			print(c)
		print()

	def plot_and_wait(self, p=None):
		""" Plot the points, and wait for the user to continue. """
		self.plot()

		if not self.wait_for_keypress:
			time.sleep(0.1)
			return

		while True:
			k = input("Press 's' to step, or 'c' to continue: ")
			if k == 's':
				break
			if k == 'c':
				self.wait_for_keypress = False
				break

	def plot(self):
		self.plot_obj.plot(self.kmeans.points, self.kmeans.centers)

	
class KMeans:
	def __init__(
		self,
		k,
		points,
		callback=None
		):

		self.num_iterations = 0
		self.points = [(p, None) for p in points]
		self.k = k
		self.callback = callback

		# The initialization right now is very simple...
		# feel free to implement a better version!
		random.shuffle(points)
		self.centers = points[:k]
		# The dimension, or the number of features, of the training 
		# examples
		self.dim = len(self.points[0])

	def loss(self):
		ans = 0
		for p, c in self.points:
			# Find the difference between the point and its center.
			diff = p - self.centers[c]
			# The norm of this vector is the distance between the two.
			ans = ans + diff.norm()
		return ans

	def run_kmeans(self):
		""" Train KMeans! """
		while True:
			self.num_iterations += 1
			old_centers = self.centers[:]
			for i in range(len(self.points)):
				p = self.points[i][0]

				# Of all the centers in self.centers, find the index
				# of the one that is closest to p. For example, 
				# if the center at index 1 is closer to p, then set
				# best_center = 1. Right now, we're just setting it
				# to something random. 
				#
				# Hint: You'll simply need to iterate through all the centers
				# (use the internet to figure out how to do this in Python, 
				# or just read this file carefully, there are many examples)
				# and keep track of which one has the smallest distance.
				# Look at the loss() function for how to find the distance 
				# between a point and a center.

				best_center = i % len(self.centers)
				self.points[i] = (p, best_center)

			# This calls the plot_and_wait function in the 2D case.
			if self.callback is not None:
				self.callback()

			# Now, calculate the new centers.
			sums = [None for i in range(len(self.centers))]
			lengths = [0]*len(self.centers)
			for x, c in self.points:
				if sums[c] is None:
					sums[c] = x
				else:
					sums[c] += x
				lengths[c] += 1

			self.centers = [None] * len(self.centers)
			for i in range(len(self.centers)):
				if lengths[i] == 0:
					self.centers[i] = old_centers[i]
				else:
					self.centers[i] = sums[i]*(1.0/lengths[i])  

			# If the centers are the same, we've converged!
			if old_centers == self.centers:
				break

			if self.callback is not None:
				self.callback()

if __name__ == "__main__":
	while True:
		k = input("Input the value of k: ")
		if k not in "123456":
			print("Must be between 1 and 6!")
		else:
			break

	p = KMeansPlot(PLOT_DIMENSION)
	examples = p.ask_for_points()
	done = False
	while not done:
		p = KMeans2D(int(k), examples)
		p.train()

		while True:
			inp = input("Press 'q' to quit, 'r' to do this again with the same points: ")
			if inp == 'q':
				done = True
				break
			elif inp == 'r':
				break
			else:
				pass

			plt.close()

