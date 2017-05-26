# Group10_MLFinalProject

Below are the approaches that we have used.  
We have two neural networks network.py and network2.py.
We have one helper utility to load minst data.

Method 1
import mnist_loader
training_data, validation_data, test_data=mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

Method 2
import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,monitor_evaluation_accuracy=True)

