import unittest
import clustering

class TestLabelConvertor(unittest.TestCase):
    def test_functionality(self):
    	labels = [0,1,0,2,2,3,4]
        self.assertSequenceEqual([len(module) for module in clustering.convert_labels2modules(labels, range(len(labels)))], [2, 1, 2, 1, 1])

    def test_noiselabels(self):
    	labels = [0,-1,0,1,1,2,2,-1]
        self.assertSequenceEqual([len(module) for module in clustering.convert_labels2modules(labels, range(len(labels)), -1)], [2,2,2])

if __name__ == '__main__':
    unittest.main()