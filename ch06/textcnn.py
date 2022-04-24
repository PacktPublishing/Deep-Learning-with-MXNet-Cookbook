import mxnet as mx

# Progress bar
from tqdm import tqdm

class TextCNN(mx.gluon.nn.Block):
    def __init__(self, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        
        # Parallel Convolution Layers
        self.parallel_convs = mx.gluon.nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.parallel_convs.add(mx.gluon.nn.Conv1D(c, k, activation='relu'))
        
        # Parallel Max-over-time Layer (can be shared, no parameters)
        self.pool = mx.gluon.nn.GlobalMaxPool1D()
        
        # Binary Classifier (Positive / Negative Review) with dropout
        self.dropout = mx.gluon.nn.Dropout(0.5)
        # No sigmoid activation function, included in loss function
        self.classifier = mx.gluon.nn.Dense(1)

    def forward(self, inputs):
        
        # Inputs are embeddings correctly formatted (batch size, embedding features, number of words)
        
        # Compute Parallel Conv Layers + Max-over-time
        # After this layer, each batch of variable-length reviews has been transformed
        # into a constant-length representation of size (batch size, sum of num-channels)
        # Please note on the operations above, that we are using the mxnet numpy module,
        # this is because these operations on mxnet ndarrays (nd module) are in place, and
        # in place operations are not supported by mxnet autograd (used later for training)
        #representation = mx.np.concatenate([
        representation = mx.nd.concatenate([
            #mx.np.squeeze(self.pool(parallel_conv(inputs)), axis=-1)
            mx.nd.squeeze(self.pool(parallel_conv(inputs)), axis=-1)
            for parallel_conv in self.parallel_convs], axis=1)

        # Classifier (with dropout)
        output = self.classifier(self.dropout(representation))
        return output
    
    def train(self, loss_fn, trainer, epochs, batch_size, training_set, validation_set, batchify_fn, ctx = mx.gpu(),
              model_file_name = "textcnn.params"):
        
        max_val_acc = 0

        # Returned values (use-case: plotting losses & validation)
        training_loss, validation_loss, validation_acc = [], [], []

        # Iterator for Gluon data access
        training_data_iterator = mx.gluon.data.DataLoader(training_set,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          batchify_fn=batchify_fn)
        
        validation_data_iterator = mx.gluon.data.DataLoader(validation_set,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            batchify_fn=batchify_fn)

        num_training_batches = len(training_set) / batch_size
        num_validation_batches = len(validation_set) / batch_size

        for e in tqdm(range(epochs)):

            cumulative_loss = 0
            # inner loop
            for data, label in training_data_iterator:
                
                # Processing data from data iterator
                data_np = data.as_np_ndarray().as_in_context(ctx)
                label_np = label.as_np_ndarray().as_in_context(ctx)
                
                with mx.autograd.record():
                    output_np = self(data_np)
                    loss = loss_fn(output_np, label_np)
                loss.backward()
                trainer.step(batch_size)
                current_loss = mx.np.mean(loss)
                cumulative_loss += current_loss / num_training_batches

            # Validation Loss & Accuracy (MXNet Metrics)
            cumulative_val_loss = 0
            val_acc = mx.metric.Accuracy()

            for data, label in validation_data_iterator:
                
                # Processing data from data iterator
                data_np = data.as_np_ndarray().as_in_context(ctx)
                label_np = label.as_np_ndarray().as_in_context(ctx)
                
                output_np = self(data_np)
                val_loss = loss_fn(output_np, label_np)
                current_val_loss = mx.np.mean(val_loss)
                cumulative_val_loss += current_val_loss / num_validation_batches

                # Accuracy
                # Comparison between labels and values output
                # Applying threshold for binary classification
                # No sigmoid necessary, as outputs of the network
                # with positive values are positive reviews
                class_output = (output_np.as_nd_ndarray() >= 0).astype("uint8").transpose()
                val_acc.update(label.as_in_context(ctx), class_output[0])

            val_acc_value = val_acc.get()[1]

            # Updating model if Validation Accuracy has improved
            added_info = ""
            if val_acc_value > max_val_acc:
                added_info = " --- Updating saved model"
                max_val_acc = val_acc_value
                self.save_parameters(model_file_name)

            desc = f"E: {e:4d}, TrL: {cumulative_loss:8.6f}, VL: {cumulative_val_loss:8.6f}, VAcc: {val_acc_value:8.6f}" + added_info
            print(desc)

            # Saving loss & accuracy values
            training_loss.append(cumulative_loss)
            validation_loss.append(cumulative_val_loss)
            validation_acc.append(val_acc_value)

        return training_loss, validation_loss, validation_acc
