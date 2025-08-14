import tensorflow as tf
import tensorflow_gnn as tfgnn


class RecurrentRelationalNetwork(tf.keras.Model):
    def __init__(self, message_dim, hidden_dim, num_steps):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

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
            tf.keras.layers.Dense(9, activation="softmax"),
        ])

    def call(self, graph: tfgnn.GraphTensor, return_all_steps=False):
        clues_one_hot = graph.node_sets["nodes"]["clues_one_hot"]
        hidden_state = graph.node_sets["nodes"]["hidden_state"]

        results = []

        for _ in range(self.num_steps):
            node_msgs = self.graph_conv(graph, edge_set_name="edges")
            messages = tf.concat([clues_one_hot, node_msgs], axis=-1)
            hidden_state, _ = self.gru_cell(messages, [hidden_state])
            graph = graph.replace_features(node_sets={"nodes": {"hidden_state": hidden_state, "clues_one_hot": clues_one_hot}})

            result = self.output_mlp(hidden_state)
            results.append(result)

        return results if return_all_steps else results[-1]
