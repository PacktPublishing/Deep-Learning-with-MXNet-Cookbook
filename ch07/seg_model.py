import mxnet as mx
import gluoncv as gcv
import numpy as np
import random
from tqdm.notebook import tqdm

# Local Libraries
import utils

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)


def training_loop(model, criterion, optimizer, epochs, batch_size, training_set, validation_set, model_filename, ctx_list):
    # Training Loop, saving best model
    # max_val_px_acc = 0
    min_val_loss = float('inf')
    
    # Returned values (use-case: plotting losses & validation)
    training_loss, training_px_acc, training_miou, validation_loss, validation_px_acc, validation_miou = [], [], [], [], [], []

    # Parallel Training
    model = gcv.utils.parallel.DataParallelModel(model, ctx_list)
    criterion = gcv.utils.parallel.DataParallelCriterion(criterion, ctx_list)

    # Iterator for Gluon data access
    training_data_iterator = mx.gluon.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        last_batch='rollover',
        num_workers=batch_size)
    
    validation_data_iterator = mx.gluon.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        last_batch='rollover',
        num_workers=batch_size)
    
    num_training_batches = len(training_set) / batch_size
    num_validation_batches = len(validation_set) / batch_size
    
    for e in tqdm(range(epochs)):
        
        # Training Loss & Accuracy
        cumulative_loss = 0.0
        segmentation_metrics = gcv.utils.metrics.segmentation.SegmentationMetric(nclass=1)
        
        # inner loop
        for data, targets in training_data_iterator:
            data = data
            targets = targets.squeeze()

            with mx.autograd.record():
                outputs = model(data)               
                losses = criterion(outputs, targets)
                mx.nd.waitall()
                mx.autograd.backward(losses)
            
            optimizer.step(batch_size)

            for loss in losses:
                current_loss = mx.nd.mean(loss)
                cumulative_loss += current_loss / num_training_batches

        #     segmentation_metrics.update(outputs, targets)
            
        # metrics_values = segmentation_metrics.get()
        # training_px_acc.append(metrics_values[0])
        # training_miou.append(metrics_values[1])
        
        # Validation Loss & Accuracy
        cumulative_val_loss = 0.0
        val_segmentation_metrics = gcv.utils.metrics.segmentation.SegmentationMetric(nclass=1)
        
        for data, label in validation_data_iterator:
            data = data
            targets = targets.squeeze()
            outputs = model(data)
            val_losses = criterion(outputs, targets)
            mx.nd.waitall()
            
            for loss in val_losses:
                current_val_loss = mx.nd.mean(loss)
                cumulative_val_loss += current_val_loss / num_training_batches
            
        #     val_segmentation_metrics.update(outputs, targets)
            
        # val_metrics_values = val_segmentation_metrics.get()
        # validation_px_acc.append(val_metrics_values[0])
        # validation_miou.append(val_metrics_values[1])
        
        # added_info = ""
        # if validation_px_acc > max_val_px_acc:
        #     added_info = " --- Updating saved model"
        #     max_val_px_acc = validation_px_acc
        #     net.save_parameters(MODEL_FILE_NAME)

        added_info = ""
        if cumulative_val_loss < min_val_loss:
            added_info = " --- Updating saved model"
            min_val_loss = cumulative_val_loss
            model.module.save_parameters(model_filename)
            
        # desc = f"E: {(e + 1):4d}, TrL: {cumulative_loss:8.6f}, TrPxAcc: {metrics_values[0]:8.6f}, VL: {cumulative_val_loss:8.6f}, VPxAcc: {val_metrics_values[0]:8.6f}" + added_info
        desc = f"E: {(e + 1):4d}, TrL: {cumulative_loss[0].asscalar():8.6f}, VL: {cumulative_val_loss[0].asscalar():8.6f}" + added_info
        print(desc)
        
        # Saving loss values
        training_loss.append(cumulative_loss)
        validation_loss.append(cumulative_val_loss)
        
    return training_loss, training_px_acc, training_miou, validation_loss, validation_px_acc, validation_miou