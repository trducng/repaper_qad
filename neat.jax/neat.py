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

from functional import evaluate, assign_species, mutate, crossover
from utils import PRNG_KEY, PRNG_SUBKEY, express, get_nodes, mean_squared_distance, draw_network


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
            each_connection['enabled'] = each_connection.get('enabled', 1)

    def express(self):
        """Express this individual into a neural network"""
        self.forward = express(self.connections)
        self.backward = express(self.connections, backward=True)


class Species():
    """A species class

    # NoneArgs
        individuals <[Individual]>: list of individuals in the species
        best_score <float>: species' best performance
        duration <int>: the time to let this species die if no improvement
    """

    def __init__(self, individuals, best_score=None, duration=6):
        global PRNG_KEY, PRNG_SUBKEY

        self.individuals = individuals
        self.role_model = individuals[random.randint(PRNG_KEY, (1,), 0, len(individuals)).item()]
        PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_KEY)
        self.best_score = best_score
        self.duration = duration

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
    global PRNG_KEY, PRNG_SUBKEY

    population = []
    innov = []
    species = []

    idx = 0
    for each_input in range(input_size):
        for each_output in range(output_size):
            innov.append({
                'from': each_input,
                'to': each_output + input_size,
                'innov': idx})
            idx += 1

    for _ in range(init_size):
        population.append(Individual(innov, input_size, output_size))

    species = assign_species(population, [], innov, [0.5, 0.4, 0.1], 0.7)
    species = [Species(value) for value in species.values()]

    for epoch in range(10):

        # evaluate
        population = []
        for each_species in species:
            diff = each_species.evaluate()

            # check if the species improves
            best = min(diff)
            if each_species.best_score is None or each_species.best_score < best:
                each_species.best_score = best
                each_species.duration = 6
            else:
                each_species.duration -= 1
            if each_species.duration <= 0:
                print("Die")
                break
            print(epoch, best)

            # get best fitness
            keep_idxs = list(
                np.argsort(np.array(diff))[:max(int(0.25*len(each_species.individuals)), 1)])

            # keep the best individuals
            keep = [each_species.individuals[idx] for idx in keep_idxs]
            n_mutate = (16 - len(keep)) // 2
            n_crossover = 16 - n_mutate - len(keep)

            new_pool = []
            new_pool += keep
            # randomly crossover
            for _ in range(n_crossover):
                base_idx_1 = random.randint(PRNG_KEY, (1,), 0, len(keep)).item()
                PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_SUBKEY)
                base_idx_2 = random.randint(PRNG_KEY, (1,), 0, len(keep)).item()
                PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_SUBKEY)

                base_ind_1 = keep[base_idx_1]
                base_ind_2 = keep[base_idx_2]

                conns = crossover(
                    base_ind_1.nodes, base_ind_1.connections,
                    base_ind_2.nodes, base_ind_2.connections, innov)
                new_pool.append(Individual(conns, input_size, output_size))

            # randomly generate the new individiuals
            for _ in range(n_mutate):
                base_idx = random.randint(PRNG_KEY, (1,), 0, len(keep)).item()
                PRNG_KEY, PRNG_SUBKEY = random.split(PRNG_SUBKEY)

                base_ind = keep[base_idx]
                nodes, connections, innov = mutate(
                    base_ind.nodes,
                    base_ind.connections,
                    innov,
                    reenable_prob=0.1,
                    connection_prob=0.3,
                    node_prob=0.3)
                new_pool.append(Individual(connections, input_size, output_size))

            population += new_pool

        species_tobe = assign_species(
            population,
            [each.role_model for each in species],
            innov,
            [0.5, 0.4, 0.1],
            0.7)
        new_species = []
        for key, value in species_tobe.items():
            if key < len(species):
                s = Species(value, best_score=species[key].best_score,
                            duration=species[key].duration)
                new_species.append(s)
            else:
                s = Species(value)
                new_species.append(s)
        species = new_species

    for idx_species, each_species in enumerate(species):
        for idx_ind, each_ind in enumerate(each_species.individuals):
            draw_network(each_ind.nodes, each_ind.connections, f'logs/{idx_species}_{idx_ind}.png')
    import pdb; pdb.set_trace()



if __name__ == "__main__":
    main(init_size=16, input_size=2, output_size=1)
