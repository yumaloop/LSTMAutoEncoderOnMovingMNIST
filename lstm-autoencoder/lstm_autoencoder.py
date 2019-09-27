import numpy as np
import tensorflow as tf


class LSTMAutoEncoder(object):
    """
    LSTM-autoencoder (cf. http://arxiv.org/abs/1502.04681)

    Args:
    =====
    - hidden_num : int
        number of hidden elements in each LSTM cell.
    - inputs : [tf.Tensor(), ...]
        list of input tensors with size (batch_num x elem_num)
    - cell : tf.python.ops.rnn_cell.LSTMCell()
        rnn cell object 
    - optimizer : tf.train.AdamOptimizer()
        optimizer for rnn
    - reverse : bool
        flag to decode in reverse order
    - decode_without_input : bool
        flag to decode without inputs
    """
    def __init__(self, 
            hidden_num, 
            inputs,
            cell=None, 
            optimizer=None,
            reverse=True,
            decode_without_input=False):

        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]
        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.optimizer = optimizer

        if cell is None:
            self.enc_cell = tf.python.ops.rnn_cell.LSTMCell(hidden_num)
            self.dec_cell = tf.python.ops.rnn_cell.LSTMCell(hidden_num)
        else:
            self.enc_cell = cell
            self.dec_cell = cell

        with tf.variable_scope('encoder'):
            (self.z_codes, self.enc_state) = tf.nn.static_rnn(self.enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num, self.elem_num], dtype=tf.float32), name='dec_weight')
            dec_bias_ = tf.Variable(tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32), name='dec_bias')

            if decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32) for _ in range(len(inputs))]
                (dec_outputs, dec_state) = tf.nn.static_rnn(self.dec_cell, dec_inputs, initial_state=self.enc_state, dtype=tf.float32)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [self.batch_num, 1, 1])
                self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
            else:
                dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
                (dec_outputs, dec_state) = ([], self.enc_state)
                for step in range(len(inputs)):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = self.dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
                    dec_outputs.append(dec_input_)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                self.output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                
        self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))
        
        if self.optimizer is None:
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        else:
            self.train_op = self.optimizer.minimize(self.loss)
        return train_op
