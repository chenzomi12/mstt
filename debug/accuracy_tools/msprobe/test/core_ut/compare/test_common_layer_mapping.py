import os
import unittest
from pathlib import Path

from msprobe.core.compare.layer_mapping import (
    generate_api_mapping_by_layer_mapping,
    generate_data_mapping_by_layer_mapping)


class TestLayerMapping(unittest.TestCase):
    def setUp(self):
        self.base_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.input_dir = os.path.join(self.base_test_dir, 'resources', 'layer_mapping')
        self.npu_dump_json = os.path.join(self.input_dir, 'mindspore', 'dump.json')
        self.bench_dump_json = os.path.join(self.input_dir, 'pytorch', 'dump.json')
        self.layer_mapping = os.path.join(self.input_dir, 'layer_mapping.yaml')

    def test_generate_api_mapping_by_layer_mapping(self):
        # Example test to check if construct.json is processed correctly
        res = generate_api_mapping_by_layer_mapping(self.npu_dump_json, self.bench_dump_json, self.layer_mapping)
        excepted_api_mapping = {
            "Cell.network_with_loss.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0":
                "Module.module.module.language_model.embedding.word_embeddings.VocabParallelEmbedding.forward.0",
        }
        self.assertDictEqual(res, excepted_api_mapping)