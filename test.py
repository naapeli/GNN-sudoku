import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


import tensorflow as tf
import pandas as pd

from model import RecurrentRelationalNetwork
from dataset import get_dataset


hidden_dim = 32  # n_node_features
message_dim = 32  # n_edge_features
num_steps = 12  # n_iters
n_nodes = 81
chunksize = 10_000

df = pd.read_csv("./sudoku9000000.csv", chunksize=chunksize)
first_chunk = next(iter(df))
n = int(0.9 * len(first_chunk))
first_chunk = first_chunk[n:]
trainset, testset = get_dataset(first_chunk, hidden_dim, batch_size=32)

model = RecurrentRelationalNetwork(message_dim, hidden_dim, num_steps)
graphs = next(iter(trainset))[0]
model(graphs)
model.load_weights("models/model_weights_end.keras")


def test_step(graphs, labels):
    logits = model(graphs)
    logits = tf.reshape(logits, [len(labels), n_nodes, 9])
    logits = tf.argmax(logits, axis=-1)

    correct = tf.math.count_nonzero(tf.cast(logits, tf.int32) == labels, axis=-1) == n_nodes
    return tf.math.count_nonzero(correct), len(correct)

total_samples = 0
total_correct = 0
for graphs, labels in testset:
    labels = labels - 1
    correct, total = test_step(graphs, labels)
    total_correct += correct
    total_samples += total

print(f"Test Acc: {100 * tf.cast(total_correct, tf.int32) / total_samples:.10f}%")
