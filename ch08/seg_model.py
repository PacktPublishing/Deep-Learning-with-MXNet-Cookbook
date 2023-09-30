import mxnet as mx
import gluoncv as gcv
import numpy as np
import random
from tqdm.notebook import tqdm
from mxnet.contrib import amp


# Disable Numpy Semantics
mx.npx.reset_np()

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)


def training_loop(model, loss_fn, trainer, epochs, batch_size, training_set, validation_set, model_filename, ctx, half_precision=False, amp_enabled=False):
    # Training Loop, saving best model
    training_loss, validation_loss = [], []
    min_val_loss = float('inf')

    # Iterators for Gluon-based optimized data access
    batch_size = 2
    training_data_iterator = mx.gluon.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, last_batch='rollover')
    validation_data_iterator = mx.gluon.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, last_batch='rollover')
    
    num_training_batches = len(training_data_iterator) / batch_size
    num_validation_batches = len(validation_data_iterator) / batch_size

    if amp_enabled:
        amp.init_trainer(trainer)

    for e in tqdm(range(epochs)):
        
        # Training Loss
        cumulative_loss = 0.0
        
        # inner loop
        for data, label in training_data_iterator:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx).squeeze()
            
            if half_precision:
                data  = data.astype('float16', copy=False)
                label = label.astype('float16', copy=False)
    
            with mx.autograd.record():
                output = model(data)            
                loss = loss_fn(output[0], label)
    
                if amp_enabled: # and half_precision:
                    with amp.scale_loss(loss, trainer) as scaled_loss:
                        mx.autograd.backward(scaled_loss)
                        # scaled_loss.backward()
                else:
                    loss.backward()
        
            trainer.step(batch_size)
            
            current_loss = mx.nd.mean(loss)            
            cumulative_loss += current_loss / num_training_batches
        
        # Validation Loss
        cumulative_val_loss = 0.0
        
        for data, label in validation_data_iterator:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx).squeeze()
            
            if half_precision:
                data = data.astype('float16', copy=False)
                label = label.astype('float16', copy=False)
            
            output = model(data)
            val_loss = loss_fn(output[0], label)
            
            current_val_loss = mx.nd.mean(val_loss)
            cumulative_val_loss += current_val_loss / num_validation_batches
    
        tr_loss = cumulative_loss.asscalar()
        val_loss = cumulative_val_loss.asscalar()
        
        added_info = ""
        if val_loss < min_val_loss:
            added_info = " --- Updating saved model"
            min_val_loss = val_loss
            model.save_parameters(model_filename)
            
        desc = f"E: {(e + 1):4d}, TrL: {tr_loss:8.6f}, VL: {val_loss:8.6f}" + added_info
        print(desc)
        
        # Saving loss values
        training_loss.append(tr_loss)
        validation_loss.append(val_loss)
        
    return training_loss, validation_loss

def multi_training_loop(model, loss_fn, trainer, epochs, batch_size, training_set, validation_set, model_filename, ctx):
    # Verify context is a list
    assert type(ctx) == list
    
    # Training Loop, saving best model
    training_loss, validation_loss = [], []
    min_val_loss = float('inf')

    # Iterators for Gluon-based optimized data access
    batch_size = 2
    training_data_iterator = mx.gluon.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, last_batch='rollover')
    validation_data_iterator = mx.gluon.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, last_batch='rollover')
    
    num_training_batches = len(training_data_iterator) / batch_size
    num_validation_batches = len(validation_data_iterator) / batch_size

    for e in tqdm(range(epochs)):
        
        # Training Loss
        cumulative_loss = 0.0
        
        # inner loop
        for data, labels in training_data_iterator:
            
            data   = mx.gluon.utils.split_and_load(data, ctx_list=ctx)
            labels = mx.gluon.utils.split_and_load(labels.squeeze(), ctx_list=ctx)
            
#             data = data.as_in_context(ctx)
#             label = label.as_in_context(ctx).squeeze()
    
            with mx.autograd.record():
#                 output = model(data)            
#                 loss = loss_fn(output[0], label)
                
                outputs = [model(data_slice) for data_slice in data]
                losses = [loss_fn(output[0], label_slice) for output, label_slice in zip(outputs, labels)]

            for loss in losses:
                loss.backward()
                current_loss = mx.nd.mean(loss)
                cumulative_loss += current_loss / num_training_batches / len(ctx)
        
            trainer.step(batch_size)
            
#             current_loss = mx.nd.mean(loss)            
#             cumulative_loss += current_loss / num_training_batches
        
        # Validation Loss
        cumulative_val_loss = 0.0
        
        for data, labels in validation_data_iterator:
#             data = data.as_in_context(ctx)
#             label = label.as_in_context(ctx).squeeze()

            data   = mx.gluon.utils.split_and_load(data, ctx_list=ctx)
            labels = mx.gluon.utils.split_and_load(labels.squeeze(), ctx_list=ctx)
            
            outputs = [model(data_slice) for data_slice in data]
            val_losses = [loss_fn(output[0], label_slice) for output, label_slice in zip(outputs, labels)]
            
            for val_loss in val_losses:
                current_val_loss = mx.nd.mean(val_loss)
                cumulative_val_loss += current_val_loss / num_validation_batches / len(ctx)
    
        tr_loss = cumulative_loss.asscalar()
        val_loss = cumulative_val_loss.asscalar()
        
        added_info = ""
        if val_loss < min_val_loss:
            added_info = " --- Updating saved model"
            min_val_loss = val_loss
            model.save_parameters(model_filename)
            
        desc = f"E: {(e + 1):4d}, TrL: {tr_loss:8.6f}, VL: {val_loss:8.6f}" + added_info
        print(desc)
        
        # Saving loss values
        training_loss.append(tr_loss)
        validation_loss.append(val_loss)
        
    return training_loss, validation_loss