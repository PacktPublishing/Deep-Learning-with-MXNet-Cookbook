import mxnet as mx

def create_regression_network():
    # MultiLayer Perceptron Model (this time using Gluon)
    net = mx.gluon.nn.Sequential()
    net.add(mx.gluon.nn.Dense(128))
    net.add(mx.gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(mx.gluon.nn.Activation('relu'))
    net.add(mx.gluon.nn.Dropout(.5))
    net.add(mx.gluon.nn.Dense(1024))
    net.add(mx.gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(mx.gluon.nn.Activation('relu'))
    net.add(mx.gluon.nn.Dropout(.4))
    net.add(mx.gluon.nn.Dense(128))
    net.add(mx.gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(mx.gluon.nn.Activation('relu'))    
    net.add(mx.gluon.nn.Dropout(.3))
    net.add(mx.gluon.nn.Dense(1))
    
    return net
