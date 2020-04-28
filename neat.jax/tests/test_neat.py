import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils import get_random_individual, draw_network
from functional import compare_genes, evaluate


class TestIndividual(unittest.TestCase):

    def test_express(self):
        """Test express phenotype from genotype"""

        inputs = [4, 8, 8, 2]
        nodes = [
            {"node_id": 0, "node_type": "input", "activation": "relu"},
            {"node_id": 1, "node_type": "input", "activation": "relu"},
            {"node_id": 2, "node_type": "input", "activation": "relu"},
            {"node_id": 3, "node_type": "input", "activation": "relu"},
            {"node_id": 4, "node_type": "output", "activation": "sigmoid"},
            {"node_id": 5, "node_type": "output", "activation": "sigmoid"},
            {"node_id": 6, "node_type": "output", "activation": "sigmoid"},
            {"node_id": 7, "node_type": "hidden", "activation": "relu"},
            {"node_id": 8, "node_type": "hidden", "activation": "relu"},
            {"node_id": 9, "node_type": "hidden", "activation": "relu"},
        ]

        connections = [
            {"from": 0, "to": 7, "weight": 0.1},
            {"from": 1, "to": 7, "weight": 0.12},
            {"from": 2, "to": 7, "weight": 0.16},
            {"from": 7, "to": 8, "weight": 0.7},
            {"from": 7, "to": 9, "weight": 0.88},
            {"from": 9, "to": 4, "weight": 0.61},
            {"from": 9, "to": 5, "weight": 0.72},
            {"from": 8, "to": 5, "weight": 0.43},
            {"from": 3, "to": 9, "weight": 0.15},
            {"from": 3, "to": 6, "weight": 0.22},
            {"from": 1, "to": 8, "weight": 0.24},
        ]

        for idx, each_connection in enumerate(connections):
            each_connection['enabled'] = 1
            each_connection['innov'] = idx

        nodes, connections = get_random_individual(10, 10, 10)
        draw_network(nodes, connections, "temp.png")
        # inputs = list(random.randint(10, size=(10,)))
        # print(calculate(inputs, nodes, connections))


class TestSpeciation(unittest.TestCase):

    def test_compare_genes(self):
        """Test excess 1"""
        gene_1 = [
            {"from": 0, "to": 7, "weight": 0.1},
            {"from": 1, "to": 7, "weight": 0.12},
            {"from": 2, "to": 7, "weight": 0.16},
            {"from": 7, "to": 8, "weight": 0.7},
            {"from": 7, "to": 9, "weight": 0.88},
            {"from": 9, "to": 4, "weight": 0.61},
            {"from": 9, "to": 5, "weight": 0.72},
            {"from": 8, "to": 5, "weight": 0.43},
            {"from": 3, "to": 9, "weight": 0.15},
            {"from": 3, "to": 6, "weight": 0.22},
            {"from": 1, "to": 8, "weight": 0.24},
            {"from": 1, "to": 9, "weight": 0.24},   # diff here
        ]

        gene_2 = [
            {"from": 0, "to": 7, "weight": 0.0},    # diff here
            {"from": 1, "to": 7, "weight": 0.12},
            {"from": 2, "to": 7, "weight": 0.16},
            {"from": 2, "to": 8, "weight": 0.16},   # diff here
            {"from": 7, "to": 8, "weight": 0.7},
            {"from": 7, "to": 9, "weight": 0.88},
            {"from": 9, "to": 4, "weight": 0.61},
            {"from": 9, "to": 5, "weight": 0.72},
            {"from": 8, "to": 5, "weight": 0.43},
            {"from": 3, "to": 9, "weight": 0.15},
            {"from": 3, "to": 6, "weight": 0.22},
            {"from": 1, "to": 8, "weight": 0.24},
            {"from": 8, "to": 4, "weight": 0.24},   # diff here
            {"from": 8, "to": 6, "weight": 0.24},   # diff here
        ]

        innov = [
            {"from": 0, "to": 7, "weight": 0.1},
            {"from": 1, "to": 7, "weight": 0.12},
            {"from": 2, "to": 7, "weight": 0.16},
            {"from": 2, "to": 8, "weight": 0.16},
            {"from": 7, "to": 8, "weight": 0.7},
            {"from": 7, "to": 9, "weight": 0.88},
            {"from": 9, "to": 4, "weight": 0.61},
            {"from": 9, "to": 5, "weight": 0.72},
            {"from": 8, "to": 5, "weight": 0.43},
            {"from": 3, "to": 9, "weight": 0.15},
            {"from": 3, "to": 6, "weight": 0.22},
            {"from": 1, "to": 8, "weight": 0.24},
            {"from": 1, "to": 9, "weight": 0.24},
            {"from": 8, "to": 4, "weight": 0.24},
            {"from": 8, "to": 6, "weight": 0.24},
        ]

        disjoint, excess, diff = compare_genes(gene_1, gene_2, innov)
        self.assertEqual(disjoint, 2)
        self.assertEqual(excess, 2)
        self.assertAlmostEqual(diff, 0.1 / 11)

class TestXOR(unittest.TestCase):

    def test_evaluate_xor_zeros(self):
        inputs = [0, 0]
        nodes = [
            {"node_id": 0, "node_type": "input", "activation": "relu"},
            {"node_id": 1, "node_type": "input", "activation": "relu"},
            {"node_id": 2, "node_type": "output", "activation": "sigmoid"},
        ]

        connections = [
            {"from": 0, "to": 2, "weight": 0.1},
            {"from": 1, "to": 2, "weight": 0.12},
        ]

        self.assertEqual(evaluate(inputs, nodes, connections)[0][2], 0)

    def test_evaluate_xor_ones(self):
        inputs = [1, 1]
        nodes = [
            {"node_id": 0, "node_type": "input", "activation": "relu"},
            {"node_id": 1, "node_type": "input", "activation": "relu"},
            {"node_id": 2, "node_type": "output", "activation": "sigmoid"},
        ]

        connections = [
            {"from": 0, "to": 2, "weight": 0.1},
            {"from": 1, "to": 2, "weight": 0.12},
        ]

        self.assertEqual(evaluate(inputs, nodes, connections)[0][2], 0.22)

    def test_evaluate_xor_zeros_ones(self):
        inputs = [0, 1]
        nodes = [
            {"node_id": 0, "node_type": "input", "activation": "relu"},
            {"node_id": 1, "node_type": "input", "activation": "relu"},
            {"node_id": 2, "node_type": "output", "activation": "sigmoid"},
        ]

        connections = [
            {"from": 0, "to": 2, "weight": 0.1},
            {"from": 1, "to": 2, "weight": 0.12},
        ]

        self.assertEqual(evaluate(inputs, nodes, connections)[0][2], 0.12)

    def test_evaluate_xor_ones_zeros(self):
        inputs = [1, 0]
        nodes = [
            {"node_id": 0, "node_type": "input", "activation": "relu"},
            {"node_id": 1, "node_type": "input", "activation": "relu"},
            {"node_id": 2, "node_type": "output", "activation": "sigmoid"},
        ]

        connections = [
            {"from": 0, "to": 2, "weight": 0.1},
            {"from": 1, "to": 2, "weight": 0.12},
        ]

        self.assertEqual(evaluate(inputs, nodes, connections)[0][2], 0.1)


if __name__ == "__main__":
    unittest.main()
