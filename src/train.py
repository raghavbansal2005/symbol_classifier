import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([1024,12,10,6])

net.SGD(training_data, 27, 25, 1.6, test_data=test_data)



