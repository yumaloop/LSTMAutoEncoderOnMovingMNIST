# LSTM-AutoEncoder on Moving-MNIST

TensorFlow LSTM-autoencoder implementation

## Usage

```python

# hidden_num : the number of hidden units in each RNN-cell
# inputs : a list of tensor with size (batch_num x step_num x elem_num)
ae = LSTMAutoEncoder(hidden_num, inputs)
...
sess.run(init)
...
sess.run(ae.train, feed_dict={input_variable:input_array,...})
```

## Reference

- Unsupervised Learning of Video Representations using LSTMs,
<br>http://arxiv.org/abs/1502.04681 (Nitish Srivastava et al., 2015)

- [Tensorflow] Building RNN Models to Solve Sequential MNIST,
<br>https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-2-f7e5ece849f5
