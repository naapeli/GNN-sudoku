import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
import pandas as pd


def edges():
    source_indices = []
    destination_indices = []
    for source in range(81):
        source_ids = []
        target_ids = []
        row = source // 9
        column = source % 9
        for target in range(9 * row, 9 * row + 9):
            if target == source or target in target_ids: continue
            source_ids.append(source)
            target_ids.append(target)
            
        for target in range(column, 81, 9):
            if target == source or target in target_ids: continue
            source_ids.append(source)
            target_ids.append(target)
        
        block_row = row // 3
        block_column = column // 3
        block_start = 9 * 3 * block_row + 3 * block_column
        for i in range(0, 3):
            for j in range(0, 3):
                target = block_start + 9 * j + i
                if target == source or target in target_ids: continue
                source_ids.append(source)
                target_ids.append(target)
        source_indices.extend(source_ids)
        destination_indices.extend(target_ids)
    return tf.convert_to_tensor(source_indices, dtype=tf.int32), tf.convert_to_tensor(destination_indices, dtype=tf.int32)


df = pd.read_csv("./sudoku10000.csv")
puzzle_array = df["puzzle"].apply(lambda x: list(map(int, x))).to_list()
puzzle_array = np.array(puzzle_array, dtype=int)
puzzle_array = puzzle_array.reshape(len(df), 9, 9)
puzzle_array = tf.convert_to_tensor(puzzle_array)
solution_array = df["solution"].apply(lambda x: list(map(int, x))).to_list()
solution_array = np.array(solution_array, dtype=int)
solution_array = solution_array.reshape(len(df), 9, 9)
solution_array = tf.convert_to_tensor(solution_array)


input_dim = 9  # n_node_inputs
hidden_dim = 16  # n_node_features
message_dim = 16  # n_edge_features
output_dim = 9  # n_node_outputs
num_steps = 8  # n_iters
n_nodes = 81


def sudoku_to_graph_tensor(puzzle, solution):
    clues = tf.reshape(puzzle, [-1])
    clues_one_hot = tf.one_hot(clues, depth=10)[:, 1:]  # only 9 features and if all of them are zero, the digit is unknown
    assert clues_one_hot.shape == (n_nodes, input_dim), "The length of clues_one_hot should be the amount of nodes (81 cells) in a graph."

    source_indices, destination_indices = edges()

    edge_features = {}
    edge_set = tfgnn.EdgeSet.from_fields(
        sizes=tf.constant([len(source_indices)]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=("nodes", source_indices),
            target=("nodes", destination_indices),
        ),
        features=edge_features
    )

    node_set = tfgnn.NodeSet.from_fields(
        sizes=tf.constant([n_nodes]),
        features={
            "clues_one_hot": clues_one_hot,
            "hidden_state": tf.zeros((n_nodes, hidden_dim))
        }
    )

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={"nodes": node_set},
        edge_sets={"edges": edge_set}
    )

    labels = tf.reshape(solution, [-1])

    return graph, labels


n = int(0.7 * len(puzzle_array))
batch_size = 32
trainset = tf.data.Dataset.from_tensor_slices((puzzle_array[:n // 10], solution_array[:n // 10]))
trainset = trainset.map(sudoku_to_graph_tensor)
trainset = trainset.batch(batch_size)
trainset = trainset.map(lambda g, y: (g.merge_batch_to_components(), y))
testset = tf.data.Dataset.from_tensor_slices((puzzle_array[n:], solution_array[n:]))
testset = testset.map(sudoku_to_graph_tensor)
testset = testset.batch(batch_size)
testset = testset.map(lambda g, y: (g.merge_batch_to_components(), y))


class RecurrentRelationalNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.message_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dense(message_dim)
        ])

        self.graph_conv = tfgnn.keras.layers.SimpleConv(
            message_fn=self.message_fn,
            receiver_tag=tfgnn.TARGET,
            combine_type="concat",
            reduce_type="sum"
        )

        self.gru_cell = tf.keras.layers.GRUCell(hidden_dim)  # input_size=input_dim + message_dim, 

        self.output_mlp = tf.keras.Sequential([
            # tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(output_dim, activation="softmax"),
        ])

    def call(self, graph: tfgnn.GraphTensor, return_all_steps=False):
        clues_one_hot = graph.node_sets["nodes"]["clues_one_hot"]
        hidden_state = graph.node_sets["nodes"]["hidden_state"]

        results = []

        for _ in range(num_steps):
            node_msgs = self.graph_conv(graph, edge_set_name="edges")
            messages = tf.concat([clues_one_hot, node_msgs], axis=-1)
            hidden_state, _ = self.gru_cell(messages, [hidden_state])
            graph = graph.replace_features(node_sets={"nodes": {"hidden_state": hidden_state, "clues_one_hot": clues_one_hot}})

            result = self.output_mlp(hidden_state)
            results.append(result)

        return results if return_all_steps else results[-1]


model = RecurrentRelationalNetwork()
graphs = next(iter(trainset))[0]
model(graphs)
model.load_weights("models/trained_sudoku_model.keras")


@tf.function
def test_step(graphs, labels):
    logits = model(graphs)
    logits = tf.reshape(logits, [len(labels), n_nodes, output_dim])
    logits = tf.argmax(logits, axis=-1)

    correct = tf.math.count_nonzero(logits == labels, axis=-1) == n_nodes
    return tf.math.count_nonzero(correct), len(correct)

total_samples = 0
total_correct = 0
for graphs, labels in testset:
    labels = labels - 1
    correct, total = test_step(graphs, labels)
    total_correct += correct
    total_samples += total

print(f"Test Acc: {tf.cast(total_correct, tf.int32) / total_samples:.10f}")
