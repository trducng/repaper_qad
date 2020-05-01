import time
from collections import defaultdict

import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import networkx as nx


PRNG_KEY = random.PRNGKey(int(time.time()))
PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)


def draw_network(nodes, connections, output_path):
    """Draw the network

    # Arguments
        nodes <[{}]>: each dict is a node
        connections <[{}]>: each dict is a connection
        output_path <str>: the path to save the graph
    """
    graph = nx.DiGraph()

    node_colors, input_nodes, output_nodes, hidden_nodes = [], [], [], []
    for each_node in nodes:
        graph.add_node(each_node["node_id"])

        if each_node["node_type"] == "input":
            node_colors.append("#b41f2e")  # red for input nodes
            input_nodes.append(each_node["node_id"])
        elif each_node["node_type"] == "output":
            node_colors.append("#1f78b4")  # blue for output nodes
            output_nodes.append(each_node["node_id"])
        else:
            node_colors.append("#b4a61f")  # yellow for hidden nodes
            hidden_nodes.append(each_node["node_id"])

    for each_connection in connections:
        if not each_connection["enabled"]:
            continue

        graph.add_edge(
            each_connection["from"],
            each_connection["to"],
            # weight=each_connection["weight"],
            # innov=each_connection["innov"],
        )

    input_pos = nx.random_layout(input_nodes)
    for k, v in input_pos.items():  # inputs are in top-left corner
        v[0] = v[0] / (1 / 0.4)
        v[1] = v[1] / (1 / 0.4) + 0.55

    output_pos = nx.random_layout(output_nodes)
    for k, v in output_pos.items():  # outputs are in bottom-left corner
        v[0] = v[0] / (1 / 0.4)
        v[1] = v[1] / (1 / 0.4)

    hidden_pos = nx.circular_layout(hidden_nodes)
    for k, v in hidden_pos.items():  # hiddens are in right side
        v[0] = min(max((v[0] + 1) / 2, 0), 1)
        v[0] = min(v[0] / (1 / 0.4) + 0.55, 0.99)
        v[1] = min(max((v[1] + 1) / 2, 0), 1)

    position = {}
    position.update(input_pos)
    position.update(output_pos)
    position.update(hidden_pos)

    # circlar
    nx.draw_networkx(
        graph,
        position,
        arrows=True,
        arrowsize=5,
        node_size=150,
        node_color=node_colors,
        font_size=6,
        style="dashed",
    )
    plt.savefig(output_path)
    plt.close()


def _get_dependency_nodes(node_id, graph):
    """Get all nodes that `node_id` relies on

    # Arguments
        node_id <int>: the id of node
        graph <{int: [int]}>: int depends on [int]

    # Returns
        <[int]>: id of nodes that `node_id` relies on
    """
    if node_id not in graph:
        return []

    if not graph[node_id]:
        return []

    result = [_ for _ in graph[node_id]]
    for each_node in graph[node_id]:
        result += _get_dependency_nodes(each_node, graph)

    return result


def express(connections, backward=False):
    """Express the connections into link format to allow faster computation

    # Args
        connections <[{}]>: each dict is a connection
        backward <bool>: if True, the returned keys are output and values are input

    # Returns
        <{int: [{}]}>: `int` is 'node_id', each {} is a node that connects to 'node_id'
    """

    graph = defaultdict(list)
    for each_node in connections:
        if backward:
            output_node = each_node["to"]
            graph[output_node].append(each_node)
        else:
            input_node = each_node["from"]
            graph[input_node].append(each_node)
    return graph


def create_random_connections(nodes, connections=[], n_new_connections=1):
    """Add new random connection from nodes and connections

    # Args
        nodes <[{}]>: each dict is a node
        connections <[{}]>: each dict is a connection
        n_new_connections <int>: number of new connections to make

    # Returns
        <[{}]>: the new connections
    """
    global PRNG_KEY, PRNG_SUBKEY

    input_pts = [idx for idx, obj in enumerate(nodes) if obj["node_type"] == "input"]
    hidden_pts = [idx for idx, obj in enumerate(nodes) if obj["node_type"] == "hidden"]
    output_pts = [idx for idx, obj in enumerate(nodes) if obj["node_type"] == "output"]

    graph = defaultdict(list)
    for each_connection in connections:
        graph[each_connection["from"]].append(each_connection["to"])

    new_connections = []

    for _ in range(n_new_connections):
        to_pts = random.shuffle(PRNG_SUBKEY, np.array(hidden_pts + output_pts))
        PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)
        for to_pt in to_pts:
            invalid_from_nodes = _get_dependency_nodes(nodes[to_pt]["node_id"], graph)
            invalid_from_nodes += [nodes[to_pt]["node_id"]]  # can't connect to itself
            invalid_from_nodes += [
                _ for _ in graph.keys() if nodes[to_pt]["node_id"] in graph[_]
            ]
            invalid_from_nodes = set(invalid_from_nodes)
            valid_from_nodes = [
                nodes[idx]["node_id"]
                for idx in input_pts + hidden_pts
                if nodes[idx]["node_id"] not in invalid_from_nodes
            ]
            if not (valid_from_nodes):
                continue

            from_node = random.shuffle(PRNG_KEY, np.array(valid_from_nodes))[0].item()
            PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)
            new_connections.append(
                {
                    "from": from_node,
                    "to": nodes[to_pt]["node_id"],
                    "weight": random.normal(PRNG_KEY).item(),
                    "enabled": 1,
                }
            )
            PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)
            graph[from_node].append(nodes[to_pt]["node_id"])
            break

    return new_connections


def get_random_individual(
    n_inputs=None, n_outputs=None, n_hiddens=None, n_connections=None
):
    global PRNG_KEY, PRNG_SUBKEY
    n_inputs = random.randint(10, 1000) if n_inputs is None else n_inputs
    n_outputs = random.randint(10, 1000) if n_outputs is None else n_outputs
    n_hiddens = random.randint(10, 1000) if n_hiddens is None else n_hiddens
    total_units = n_inputs + n_outputs + n_hiddens
    n_connections = (
        random.randint(PRNG_KEY, (1,), total_units, total_units * 5).item()
        if n_connections is None
        else n_connections
    )
    PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)

    nodes = []
    for _ in range(n_inputs):
        nodes.append(
            {"node_id": len(nodes), "node_type": "input", "activation": "relu"}
        )
    for _ in range(n_outputs):
        nodes.append(
            {"node_id": len(nodes), "node_type": "output", "activation": "sigmoid"}
        )
    for _ in range(n_hiddens):
        nodes.append(
            {"node_id": len(nodes), "node_type": "hidden", "activation": "relu"}
        )

    connections = create_random_connections(nodes, n_new_connections=n_connections)
    print(n_connections)
    print(len(connections))

    return nodes, connections


def xor(a, b):
    """The XOR problem

    # Args
        a <int>: 0 or 1
        b <int>: 0 or 1

    # Returns
        <int> whether the condition results in 0 or 1
    """
    if a not in [0, 1] or b not in [0, 1]:
        raise AttributeError("`a` and `b` should be 0 or 1")

    return int((a + b) == 1)


def get_nodes(connections, input_size, output_size):
    """Get nodes from connections

    # Args
        connections <[{}]>: each dict is a connection
        input_size <int>: the amount of input units
        output_size <int>: the amount of output units

    # Returns
        nodes <[{}]>: each dict is a node
    """
    nodes = list(range(input_size + output_size))
    for each_connection in connections:
        nodes += [each_connection["from"], each_connection["to"]]
    nodes = sorted(list(set(nodes)))

    result = []
    for each_node in nodes:
        if each_node < input_size:
            ntype = "input"
            act = "none"
        elif each_node < input_size + output_size:
            ntype = "output"
            act = "sigmoid"
        else:
            ntype = "hidden"
            act = "relu"

        result.append({"node_id": each_node, "node_type": ntype, "activation": act})

    return result

def mean_squared_distance(predictions, ground_truths):
    diff = 0
    for pred, ground in zip(predictions, ground_truths):
        diff += (pred - ground) ** 2
    return diff / len(predictions)
