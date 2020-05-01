from collections import defaultdict
from copy import deepcopy

import jax.numpy as np
import jax.random as random

from utils import PRNG_KEY, PRNG_SUBKEY, express, create_random_connections
from nn_ops import relu, sigmoid



def get_node_value(node_id, phenotype, cache, inputs):
    """Calculate node value recursively

    # Arguments
        node_id <int>: the node id
        phenotype <{int: [{}]}>: int = node_id of output, list = input nodes
        cache <{int: float}>: store computed node value to avoid duplicating calculation
        inputs [list of ints]: the value of input nodes

    # Returns
        [float]: the value of node
    """
    if node_id not in phenotype:
        if node_id < len(inputs):  # means this is input node
            return inputs[node_id]
        else:  # means this is orphan hidden node
            return 0

    total = 0
    for each_input_connection in phenotype[node_id]:
        input_node_id = each_input_connection["from"]
        if input_node_id in cache:
            total += cache[input_node_id] * each_input_connection["weight"]
            continue

        value = get_node_value(input_node_id, phenotype, cache, inputs)
        value = sigmoid(value)
        total += value * each_input_connection["weight"]
        cache[input_node_id] = value

    return total


def evaluate(inputs, nodes, connections):
    """Calculate the output values of a network

    # Args
        inputs <[int]>: the real input values
        nodes <[{}]>: each dict is a node
        connections <[{}]>: each dict is a connection

    # Returns
        <[float]>: each float is value of an output neuron
    """

    phenotype = express(connections, backward=True)
    output_nodes = [node for node in nodes if node["node_type"] == "output"]
    cache = {}
    result = {}
    for each_node in output_nodes:
        value = sigmoid(
            get_node_value(each_node["node_id"], phenotype, cache, inputs))
        result[each_node["node_id"]] = value
    return result


def compare_genes(connections_a, connections_b, innov, crossover=False):
    """Compare 2 genes to get excess, disjoint and weight difference

    # Args
        connections_a <[{}]>: each dict is a connection
        connections_b <[{}]>: each dict is a connection
        innov <[{}]>: each dict is a connection

    # Returns
        <int>: total number of disjoints
        <int>: total number of excesses
        <float>: average weight difference
    """
    indices_a, indices_b = [], []
    innov_a, innov_b = [], []
    idx_a, idx_b = 0, 0
    for idx, each_connection in enumerate(innov):
        f, t = each_connection["from"], each_connection["to"]

        if idx_a < len(connections_a):
            current_a = connections_a[idx_a]
            if current_a["from"] == f and current_a["to"] == t:
                indices_a.append(idx_a)
                innov_a.append(idx)
                idx_a += 1

        if idx_b < len(connections_b):
            current_b = connections_b[idx_b]
            if current_b["from"] == f and current_b["to"] == t:
                indices_b.append(idx_b)
                innov_b.append(idx)
                idx_b += 1

    if crossover:
        return indices_a, innov_a, indices_b, innov_b

    # indices_a & indices_b must point to `innov` by now
    commons = set(innov_a).intersection(set(innov_b))
    total_weight_diff = 0.0
    for each_innov in sorted(list(commons)):
        idx_a = innov_a.index(each_innov)
        idx_b = innov_b.index(each_innov)

        total_weight_diff += abs(
            connections_a[indices_a[idx_a]]['weight'] - connections_b[indices_b[idx_b]]['weight']
        )
    weight_diff = total_weight_diff / len(commons)

    disjoint_a = sorted(list(set(innov_a).difference(commons)))
    disjoint_b = sorted(list(set(innov_b).difference(commons)))

    max_a, max_b = innov_a[-1], innov_b[-1]
    if max_a > max_b:
        excess = [_ for _ in disjoint_a if _ > max_b]
    elif max_a < max_b:
        excess = [_ for _ in disjoint_b if _ > max_a]
    else:
        excess = []

    n_excess = len(excess)
    n_disjoint = len(disjoint_a) + len(disjoint_b) - n_excess

    return n_disjoint, n_excess, weight_diff


def assign_species(population, species, innov, alpha, thresh):
    """Assign species

    # Args
        population <[Individual]>: list of Individuals
        species <[Individual]>: list of role model Individals

    # Returns
        <{int: []}>: key is the species' index, value is the individuals
    """
    result = defaultdict(list)

    for each_individual in population:
        for idx, each_species in enumerate(species):
            n_disjoint, n_excess, weight_diff = compare_genes(
                each_individual.connections, each_species.connections, innov)
            n = max(len(each_individual.connections), len(each_species.connections))
            if n_disjoint * alpha[0] / n + n_excess * alpha[1] / n + weight_diff * alpha[2] < thresh:
                result[idx].append(each_individual)
                break
        else:
            result[len(species)].append(each_individual)
            species += [each_individual]

    return result


def crossover(nodes_a, conns_a, nodes_b, conns_b, innov):
    """Perform crossover between individuals

    # Args
        nodes_a <{}>: each dict is a node
        conns_a <{}>: each_dict is a connection
        nodes_b <{}>: each dict is a node
        conns_b <{}>: each_dict is a connection
        innov <[{}]>: each dict is a connection

    # Returns
        <{}>: new connections
    """
    global PRNG_KEY, PRNG_SUBKEY

    indices_a, innov_a, indices_b, innov_b = compare_genes(conns_a, conns_b, innov, True)
    commons = set(innov_a).intersection(set(innov_b))
    innov_indices = sorted(list(set(innov_a + innov_b)))


    result = []
    for idx in innov_indices:
        if idx in commons:
            r = random.normal(PRNG_KEY).item()
            if r > 0.5:
                conn = conns_a[indices_a[innov_a.index(idx)]]
            else:
                conn = conns_b[indices_b[innov_b.index(idx)]]
            PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_SUBKEY)
            result.append(conn)
            continue

        if idx in innov_a:
            conn = conns_a[indices_a[innov_a.index(idx)]]
            result.append(conn)

        if idx in innov_b:
            conn = conns_b[indices_b[innov_b.index(idx)]]
            result.append(conn)

    return result


def mutate(nodes, connections, innovation, reenable_prob, connection_prob, node_prob):
    """Mutate a single individual

    Mutation rules:
        - re-enabled weights
        - add conections
        - add nodes

    # Args
        nodes <[{}]>: each dict is a node
        connections <[{}]>: each dict is a connection
        innovation  <[{}]>: each dict is a connection record
        reenable_prob <float>: probability that weights are re-enabled
        connection_prob <float>: probability to use add_connection mutation
        node_prob <float>: probabilityto use add node mutation

    # Returns
        <[{}]>: new nodes list
        <[{}]>: new connections list
        <[{}]>: new innovation list
    """
    global PRNG_KEY, PRNG_SUBKEY

    # flip weight from disabled to enabled
    for each_connection in connections:
        if each_connection["enabled"]:
            continue

        if random.normal(PRNG_SUBKEY).item() < reenable_prob:
            each_connection["enabled"] = 1
            PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)

    if random.normal(PRNG_KEY).item() < connection_prob:
        connections, innovation = mutate_add_new_connections(
            nodes, connections, innovation
        )
    PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_SUBKEY)

    if random.normal(PRNG_KEY).item() < node_prob:
        nodes, connections, innovation = mutate_add_new_node(
            nodes, connections, innovation
        )
    PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_SUBKEY)

    return nodes, connections, innovation


def mutate_add_new_connections(nodes, connections, innovation):
    """Mutate a single individual by adding new connections

    Adding rule:  between 2 nodes that are not connected according to `innovation`,
    weights are random. The connection should not create loop.

    # Args
        nodes <[{}]>: each dict is a node
        connections <[{}]>: each dict is a connection
        innovation  <[{}]>: each dict is a connection record

    # Returns
        <[{}]>: new connections list
        <[{}]>: new innovation list
    """
    new_connections = create_random_connections(nodes, innovation, n_new_connections=1)
    connections += new_connections

    new_connections = deepcopy(new_connections)
    for idx, each in enumerate(new_connections):
        del each["weight"]
        del each['enabled']
        each["innov"] = innovation[-1]['innov'] + idx + 1
    innovation = innovation + new_connections

    return connections, innovation


def mutate_add_new_node(nodes, connections, innovation):
    """Mutate a single individual by adding new node

    # Args
        nodes <[{}]>: each dict is a node
        connections <[{}]>: each dict is a connection
        innovation  <[{}]>: each dict is a connection record

    # Returns
        <[{}]>: each dict is a node
        <[{}]>: new connections list
        <[{}]>: new innovation list
    """
    global PRNG_KEY, PRNG_SUBKEY

    node_ids = []
    for each_connection in innovation:
        node_ids.append(each_connection["from"])
        node_ids.append(each_connection["to"])
    node_ids = list(set(node_ids))

    new_id = max(node_ids) + 1
    replaced_connection = random.shuffle(
        PRNG_KEY, np.array(list(range(len(connections))))
    )
    PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_SUBKEY)
    for idx in replaced_connection:
        connection = connections[idx]
        if not (connection["enabled"]):
            continue
        connections.append(
            {"from": connection["from"], "to": new_id, "weight": 1, "enabled": 1}
        )
        connections.append(
            {
                "from": new_id,
                "to": connection["to"],
                "weight": connection["weight"],
                "enabled": 1,
            }
        )
        connections[idx]["enabled"] = 0

        innovation.append({"from": connection["from"], "to": new_id, "innov": len(innovation)})
        innovation.append({"from": new_id, "to": connection["to"], "innov": len(innovation)})
        break

    nodes.append({"node_id": new_id, "node_type": "hidden", "activation": "relu"})

    return nodes, connections, innovation
