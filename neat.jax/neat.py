"""
Processes

1. Start with a pool, have a single gene in that pool
2. Do evaluation
3. Select the k% best genes
3a. Copy them to the new pool
3b. Create crossover of these genese and add to the new pool
3c. Create mutation these genes and add to the new pool
4. For mutation that introduces new topology, use speciation function to group the genese into species, perform evolution on these groups
    - Basically, force individuals with similar genomes to share their fitness payoffs.

Steps

1. Define the genotypes and historical marking [done]
2. Construct speciation function
3. Construct evaluate function
4. Construct permutation and crossover function
5. Construct loading the dataset
"""
from copy import deepcopy

import jax.numpy as np
import jax.random as random
from datasource import download
from datasource.loaders import mnist_loader

from functional import evaluate, assign_species
from utils import PRNG_KEY, PRNG_SUBKEY, express, get_nodes, mean_squared_distance


xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs = [0, 1, 1, 0]


class Individual():
    """An individual living in the population and in the species

    # Args
        nodes <[{}]>: each dict is a node (node_id, node_type, activation)
        connections <[{}]>: each dict is a connection (input_node, output_node,
            weight, enabled)
    """

    def __init__(self, connections, input_size, output_size):
        """Initialize the individual"""
        self.input = input_size
        self.output = output_size
        self.nodes = get_nodes(connections, input_size, output_size)
        self.connections = deepcopy(connections)

        # phenotype representation
        self.forward, self.backward = None, None

        self.initialize_individual()
        self.express()

    def initialize_individual(self):
        """Initialize the connections.

        Initializing an individual basically just jitter the weight values
        """
        global PRNG_KEY, PRNG_SUBKEY

        for each_connection in self.connections:
            each_connection['weight'] = random.normal(PRNG_KEY).item()
            PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)

    def express(self):
        """Express this individual into a neural network"""
        self.forward = express(self.connections)
        self.backward = express(self.connections, backward=True)


class Species():
    """A species class

    # Args
        individuals <[Individual]>: list of individuals in the species
        best_score <float>: species' best performance
        duration <int>: the time to let this species die if no improvement
    """

    def __init__(self, individuals, best_score=None, duration=5):
        global PRNG_KEY, PRNG_SUBKEY

        self.individuals = individuals
        self.role_model = individuals[random.randint(PRNG_KEY, (1,), 0, len(individuals)).item()]
        PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)
        self.best_score = best_score
        self.duration = None

    def evaluate(self):
        diff = []
        for each_individual in self.individuals:
            preds = [
                list(evaluate([x1, x2], each_individual.nodes, each_individual.connections)[0].values())[0]
                for x1, x2 in xor_inputs
            ]
            diff.append(mean_squared_distance(preds, xor_outputs))
        return diff



def main(init_size, input_size, output_size):
    population = []
    innov = []
    species = []

    idx = 0
    for each_input in range(input_size):
        for each_output in range(output_size):
            innov.append({'from': each_input, 'to': each_output + input_size, 'innov': idx})
            idx += 1

    for _ in range(init_size):
        population.append(Individual(innov, input_size, output_size))

    species = assign_species(population, [], innov, [0.5, 0.4, 0.1], 0.2)
    species = [Species(value) for value in species.values()]

    for epoch in range(2):

        # evaluate
        for each_species in species:
            diff = each_species.evaluate()

            # get best fitness
            keep_idxs = list(
                np.argsort(np.array(diff))[:max(int(0.25*len(each_species.individuals)), 1)])

            # give birth
            new_pool = [each_species[idx] for idx in keep_idxs]



        # reassign species


if __name__ == "__main__":
    main(init_size=16, input_size=2, output_size=1)
