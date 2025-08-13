import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf
import pandas as pd

from model import RecurrentRelationalNetwork
from dataset import get_dataset


hidden_dim = 16  # n_node_features
message_dim = 16  # n_edge_features
num_steps = 8  # n_iters
n_nodes = 81

model = RecurrentRelationalNetwork(message_dim, hidden_dim, num_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

train_loss_metric = tf.keras.metrics.Mean(name="train_loss")

@tf.function
def train_step(graphs, labels):
    with tf.GradientTape() as tape:
        results = model(graphs, return_all_steps=True)

        total_loss = 0
        for logits in results:
            logits = tf.reshape(logits, [len(labels), n_nodes, 9])
            loss = criterion(labels, logits)
            total_loss += loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss_metric.update_state(total_loss)

@tf.function
def test_step(graphs, labels):
    logits = model(graphs)
    logits = tf.reshape(logits, [len(labels), n_nodes, 9])
    logits = tf.argmax(logits, axis=-1)

    correct = tf.math.count_nonzero(tf.cast(logits, tf.int32) == labels, axis=-1) == n_nodes
    return tf.math.count_nonzero(correct), len(correct)


chunksize = 1000  # 1_000_000

epochs = 2
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} / {epochs}")

    chunks = pd.read_csv("./sudoku9000000.csv", chunksize=chunksize)
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i + 1} / {9_000_000 // chunksize}")

        trainset, testset = get_dataset(chunk, hidden_dim)

        train_loss_metric.reset_states()

        for graphs, labels in trainset:
            labels = labels - 1
            train_step(graphs, labels)

        total_samples = 0
        total_correct = 0
        for graphs, labels in testset:
            labels = labels - 1
            correct, total = test_step(graphs, labels)
            total_correct += correct
            total_samples += total

        print(f"        Train Loss: {train_loss_metric.result():.4f}")
        print(f"        Test Acc: {100 * tf.cast(total_correct, tf.int32) / total_samples:.10f}%")

model.save("models/trained_sudoku_model.keras")
