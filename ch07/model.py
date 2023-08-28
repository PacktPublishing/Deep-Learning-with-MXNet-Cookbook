import mxnet as mx
from mxnet.gluon import nn
import numpy as np
import random

# Numpy Semantics
mx.npx.set_np()

# Local Libraries
import utils

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)

MODEL_FILE_NAME = "resnet50.params"
CLASSES_DICT = utils.generate_class_dict_cats_vs_dogs_imagenet()

def training_loop(net, loss_fn, trainer, epochs, batch_size, training_set, validation_set, ctx = mx.gpu()):
    # Training Loop, saving best model
    max_val_acc = 0
    
    # Returned values (use-case: plotting losses & validation)
    training_loss, training_acc, validation_loss, validation_acc = [], [], [], []
    
    # Iterator for Gluon data access
    training_data_iterator = mx.gluon.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_data_iterator = mx.gluon.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    
    num_training_batches = len(training_set) / batch_size
    num_validation_batches = len(validation_set) / batch_size
    
    for e in range(epochs):
        
        # Training Loss & Accuracy
        cumulative_loss = 0.0
        train_acc = mx.metric.Accuracy()
        
        # inner loop
        for data, label in training_data_iterator:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with mx.autograd.record():
                output = net(data)
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)
            
            current_loss = mx.np.mean(loss)            
            cumulative_loss += current_loss / num_training_batches

            # No need to compute softmax output,
            # the class with the larger value is the selected one
            # softmax_output = mx.nd.softmax(output.as_nd_ndarray())
            class_output = np.argmax(output, axis=1)    
            train_acc.update(label, class_output)
            
        tr_acc_value = train_acc.get()[1]
        training_acc.append(tr_acc_value)
        
        # Validation Loss & Accuracy
        cumulative_val_loss = 0.0
        val_acc = mx.metric.Accuracy()
        
        for data, label in validation_data_iterator:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = net(data)
            
            val_loss = loss_fn(output, label)
            
            current_val_loss = mx.np.mean(val_loss)
            cumulative_val_loss += current_val_loss / num_validation_batches
            
            # No need to compute softmax output,
            # the class with the larger value is the selected one
            # softmax_output = mx.nd.softmax(output.as_nd_ndarray())
            class_output = np.argmax(output, axis=1) 
            val_acc.update(label, class_output)
            
        val_acc_value = val_acc.get()[1]
        validation_acc.append(val_acc_value)
        
        added_info = ""
        if val_acc_value > max_val_acc:
            added_info = " --- Updating saved model"
            max_val_acc = val_acc_value
            net.save_parameters(MODEL_FILE_NAME)
            
        desc = f"E: {(e + 1):4d}, TrL: {cumulative_loss:8.6f}, TrAcc: {tr_acc_value:8.6f}, VL: {cumulative_val_loss:8.6f}, VAcc: {val_acc_value:8.6f}" + added_info
        print(desc)
        
        # Saving loss values
        training_loss.append(cumulative_loss)
        validation_loss.append(cumulative_val_loss)
        
    return training_loss, training_acc, validation_loss, validation_acc