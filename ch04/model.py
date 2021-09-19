import mxnet as mx

def create_classification_network(num_outputs = 3):
    # MLP with Gluon
    net = mx.gluon.nn.Sequential()
    net.add(mx.gluon.nn.Dense(10, activation="relu"))
    net.add(mx.gluon.nn.Dense(10, activation="relu"))
    net.add(mx.gluon.nn.Dense(num_outputs))
    # Note that the latest layer does not have an activation
    # function whereas Softmax was expected.
    # This is due to an optimization during training:
    # the loss function includes the softmax computation.
    return net
