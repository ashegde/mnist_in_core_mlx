"""
In this script, we construct a simple MLP using just core MLX.
The resulting MLP is then trained on MNIST. This example is 
akin to https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html,
with some modifications for fun.
"""
import time
import mlx.core as mx
import numpy as np

from mnist import mnist
from model import build_params, predict_single

mx.random.seed(2025)

layer_sizes = [784, 512, 512, 10]
params = build_params(layer_sizes=layer_sizes)
batched_predict = mx.vmap(predict_single, in_axes=(None, 0))

train_x, train_y, test_x, test_y = map(mx.array, mnist())

# loss function and metrics

def accuracy(
        params: list[mx.array],
        images: mx.array,
        targets: mx.array,
) -> mx.array:
    predicted_class = mx.argmax(batched_predict(params, images), axis=-1)
    return mx.mean(predicted_class == targets)

def cross_entropy_loss(
        params: list[mx.array],
        images: mx.array,
        targets: mx.array,
) -> mx.array:
    preds = batched_predict(params, images)
    # recall, preds returns log(softmax probs) as rows
    # whereas targets contains the target class index
    return -mx.mean(preds[mx.arange(targets.size), targets])

# model training
learning_rate = 0.01
num_epochs = 10
batch_size = 128

loss_and_grad_fn = mx.value_and_grad(cross_entropy_loss, argnums=0)

def batch_iterate(
        batch_size: int,
        X: mx.array,
        y: mx.array,
):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


for epoch in range(num_epochs):
    start_time = time.time()
    epoch_loss = []
    for xb, yb in batch_iterate(batch_size, train_x, train_y):
        batch_loss, batch_grad = loss_and_grad_fn(params, xb, yb)

        # parameter update via SGD
        params = [
            (
            p[0] - learning_rate * g[0],
            p[1] - learning_rate * g[1],
            ) for p, g in zip(params, batch_grad)
        ]
        epoch_loss.append(batch_loss.item())

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    epoch_time = time.time() - start_time
    train_acc = accuracy(params, train_x, train_y)
    test_acc = accuracy(params, test_x, test_y)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Average training loss {}".format(avg_loss))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))

