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
            self.enc_cell = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=3, strides=(1,1), padding="valid", data_format="channels_last")
            self.dec_cell = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=3, strides=(1,1), padding="valid", data_format="channels_last")
        else:
            self.enc_cell = cell
            self.dec_cell = cell

        with tf.variable_scope('encoder'):
            encoder = tf.keras.layers.RNN(self.enc_cell)
            z_codes = encoder(inputs)

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.random.truncated_normal([hidden_num, self.elem_num], dtype=tf.float32), name='dec_weight')
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
            self.train_op = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)
            # self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        else:
            self.train_op = self.optimizer.minimize(self.loss)


class ConvLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    '''Convolutional LSTM (Long short-term memory unit) recurrent network cell.
    The class uses optional peep-hole connections, optional cell-clipping,
    optional normalization layer, and an optional recurrent dropout layer.
    Basic implmentation is based on tensorflow, tf.nn.rnn_cell.LSTMCell.
    Default LSTM Network implementation is based on:
        http://www.bioinf.jku.at/publications/older/2604.pdf
    Sepp Hochreiter, Jurgen Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
    Peephole connection is based on:
        https://research.google.com/pubs/archive/43905.pdf
    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for large scale acoustic modeling". 2014.
    Default Convolutional LSTM implementation is based on:
        https://arxiv.org/abs/1506.04214
    Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo.
    "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting". 2015.
    Recurrent dropout is base on:
    
        https://arxiv.org/pdf/1603.05118
    Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth
    "Recurrent Dropout without Memory Loss". 2016.
    Normalization layer is applied prior to nonlinearities.
    '''
    def __init__(self,
                 shape,
                 kernel,
                 depth,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 forget_bias=1.0,
                 activation=None,
                 normalize=None,
                 dropout=None,
                 reuse=None):
        '''Initialize the parameters for a ConvLSTM Cell.
        Args:
            shape: list of 2 integers, specifying the height and width 
                of the input tensor.
            kernel: list of 2 integers, specifying the height and width 
                of the convolutional window.
            depth: Integer, the dimensionality of the output space.
            use_peepholes: Boolean, set True to enable diagonal/peephole connections.
            cell_clip: Float, if provided the cell state is clipped by this value 
                prior to the cell output activation.
            initializer: The initializer to use for the weights.
            forget_bias: Biases of the forget gate are initialized by default to 1
                in order to reduce the scale of forgetting at the beginning of the training.
            activation: Activation function of the inner states. Default: `tanh`.
            normalize: Normalize function, if provided inner states is normalizeed 
                by this function.
            dropout: Float, if provided dropout is applied to inner states 
                with keep probability in this value.
            reuse: Boolean, whether to reuse variables in an existing scope.
        '''
        super(ConvLSTMCell, self).__init__(_reuse=reuse)

        tf_shape = tf.TensorShape(shape + [depth])
        self._output_size = tf_shape
        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(tf_shape, tf_shape)

        self._kernel = kernel
        self._depth = depth
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation or tf.nn.tanh
        self._normalize = normalize
        self._dropout = dropout

        self._w_conv = None
        if self._use_peepholes:
            self._w_f_diag = None
            self._w_i_diag = None
            self._w_o_diag = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        '''Run one step of ConvLSTM.
        Args:
            inputs: input Tensor, 4D, (batch, shape[0], shape[1], depth)
            state: tuple of state Tensor, both `4-D`, with tensor shape `c_state` and `m_state`.
        Returns:
            A tuple containing:
            - A '4-D, (batch, height, width, depth)', Tensor representing 
                the output of the ConvLSTM after reading `inputs` when previous 
                state was `state`.
                Here height, width is:
                    shape[0] and shape[1].
            - Tensor(s) representing the new state of ConvLSTM after reading `inputs` when
                the previous state was `state`. Same type and shape(s) as `state`.
        '''
        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(4)[3]
        if input_size.value is None:
            raise ValueError('Could not infer size from inputs.get_shape()[-1]')

        c_prev, m_prev = state
        inputs = tf.concat([inputs, m_prev], axis=-1)

        if not self._w_conv:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, initializer=self._initializer):
                kernel_shape = self._kernel + [inputs.shape[-1].value, 4 * self._depth]
                self._w_conv = tf.get_variable('w_conv', shape=kernel_shape, dtype=dtype)

        # i = input_gate, j = new_input, f = forget_gate, o = ouput_gate
        conv = tf.nn.conv2d(inputs, self._w_conv, (1, 1, 1, 1), 'SAME')
        i, j, f, o = tf.split(conv, 4, axis=-1)

        # Diagonal connections
        if self._use_peepholes and not self._w_f_diag:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, initializer=self._initializer):
                self._w_f_diag = tf.get_variable('w_f_diag', c_prev.shape[1:], dtype=dtype)
                self._w_i_diag = tf.get_variable('w_i_diag', c_prev.shape[1:], dtype=dtype)
                self._w_o_diag = tf.get_variable('w_o_diag', c_prev.shape[1:], dtype=dtype)

        if self._use_peepholes:
            f = f + self._w_f_diag * c_prev
            i = i + self._w_i_diag * c_prev
        if self._normalize is not None:
            f = self._normalize(f)
            i = self._normalize(i)
            j = self._normalize(j)

        j = self._activation(j)

        if self._dropout is not None:
            j = tf.nn.dropout(j, self._dropout)

        c = tf.nn.sigmoid(f + self._forget_bias) * c_prev + tf.nn.sigmoid(i) * j

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            o = o + self._w_o_diag * c
        if self._normalize is not None:
            o = self._normalize(o)
            c = self._normalize(c)

        m = tf.nn.sigmoid(o) * self._activation(c)

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, m)
        return m, new_state
