import os
import re
import json
import unittest
import shutil
import argparse
from dataclasses import dataclass

from unittest.mock import patch
from msprobe.visualization.graph_service import _compare_graph_result, _build_graph_result, _compare_graph_ranks, \
    _compare_graph_steps, _build_graph_ranks, _build_graph_steps, _graph_service_command, _graph_service_parser
from msprobe.core.common.utils import CompareException


@dataclass
class Args:
    input_path: str = None
    output_path: str = None
    layer_mapping: str = None
    framework: str = None
    overflow_check: bool = False
    fuzzy_match: bool = False
    complete_stack: bool = False
    parallel_merge: bool = False
    parallel_params: tuple = None


class TestGraphService(unittest.TestCase):
    def setUp(self):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.input = os.path.join(self.current_path, "input_format_correct")
        self.output = os.path.join(self.current_path, 'output')
        self.input_param = {
            'npu_path': os.path.join(self.input, 'step0', 'rank0'),
            'bench_path': os.path.join(self.input, 'step0', 'rank0'),
            'is_print_compare_log': True
        }
        self.layer_mapping = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'layer_mapping.yaml')
        self.pattern = r'\b\w+\.vis\b'
        self.pattern_rank = r'[\w_]+\.vis\b'
        self.output_json = []
        for i in range(7):
            self.output_json.append(os.path.join(self.current_path, f"compare{i}.json"))

    def assert_log_info(self, mock_log_info,
                        log_info='Model graphs compared successfully, the result file is saved in'):
        last_call_args = mock_log_info.call_args[0][0]
        self.assertIn(log_info, last_call_args)
        matches = re.findall(self.pattern, last_call_args)
        if matches:
            self.assertTrue(os.path.exists(os.path.join(self.output, matches[0])))

    @patch('msprobe.core.common.log.logger.info')
    def test_compare_graph_result(self, mock_log_info):
        args = Args(output_path=self.output, framework='pytorch')
        result = _compare_graph_result(self.input_param, args)
        self.assertEqual(mock_log_info.call_count, 2)
        self.assertIsNotNone(result)

        args = Args(output_path=self.output, framework='mindspore')
        result = _compare_graph_result(self.input_param, args)
        self.assertIsNotNone(result)

        args = Args(output_path=self.output, framework='pytorch', layer_mapping=self.layer_mapping)
        result = _compare_graph_result(self.input_param, args)
        self.assertIsNotNone(result)

        args = Args(output_path=self.output, framework='pytorch', overflow_check=True)
        result = _compare_graph_result(self.input_param, args)
        self.assertIsNotNone(result)

    @patch('msprobe.core.common.log.logger.info')
    def test_build_graph_result(self, mock_log_info):
        result = _build_graph_result(os.path.join(self.input, 'step0', 'rank0'), Args(overflow_check=True))
        self.assertEqual(mock_log_info.call_count, 1)
        self.assertIsNotNone(result)

    @patch('msprobe.core.common.log.logger.info')
    def test_compare_graph_ranks(self, mock_log_info):
        input_param = {
            'npu_path': os.path.join(self.input, 'step0'),
            'bench_path': os.path.join(self.input, 'step0'),
            'is_print_compare_log': True
        }
        args = Args(output_path=self.output, framework='pytorch')
        _compare_graph_ranks(input_param, args)
        self.assert_log_info(mock_log_info, 'Successfully exported compare graph results.')

        input_param1 = {
            'npu_path': os.path.join(self.input, 'step0'),
            'bench_path': os.path.join(self.input, 'step1'),
            'is_print_compare_log': True
        }
        args = Args(output_path=self.output, framework='pytorch')
        with self.assertRaises(CompareException):
            _compare_graph_ranks(input_param1, args)

    @patch('msprobe.core.common.log.logger.info')
    def test_compare_graph_steps(self, mock_log_info):
        input_param = {
            'npu_path': self.input,
            'bench_path': self.input,
            'is_print_compare_log': True
        }
        args = Args(output_path=self.output, framework='pytorch')
        _compare_graph_steps(input_param, args)
        self.assert_log_info(mock_log_info, 'Successfully exported compare graph results.')

        input_param1 = {
            'npu_path': self.input,
            'bench_path': os.path.join(self.current_path, "input"),
            'is_print_compare_log': True
        }
        args = Args(output_path=self.output, framework='pytorch')
        with self.assertRaises(CompareException):
            _compare_graph_steps(input_param1, args)

    @patch('msprobe.core.common.log.logger.info')
    def test_build_graph_ranks(self, mock_log_info):
        _build_graph_ranks(os.path.join(self.input, 'step0'), Args(output_path=self.output))
        self.assert_log_info(mock_log_info, "Successfully exported build graph results.")

    @patch('msprobe.core.common.log.logger.info')
    def test_build_graph_steps(self, mock_log_info):
        _build_graph_steps(self.input, Args(output_path=self.output))
        self.assert_log_info(mock_log_info, "Successfully exported build graph results.")

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command(self, mock_log_info):
        with open(self.output_json[0], 'w') as f:
            json.dump(self.input_param, f, indent=4)

        args = Args(input_path=self.output_json[0], output_path=self.output, framework='pytorch')
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, 'Exporting compare graph result successfully, the result file is saved in')

        input_param1 = {
            'npu_path': os.path.join(self.input, 'step0', 'rank0'),
            'is_print_compare_log': True
        }
        with open(self.output_json[1], 'w') as f:
            json.dump(input_param1, f, indent=4)
        args = Args(input_path=self.output_json[1], output_path=self.output, framework='pytorch')
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, "Model graph exported successfully, the result file is saved in")

        input_param2 = {
            'npu_path': os.path.join(self.input, 'step0'),
            'bench_path': os.path.join(self.input, 'step0'),
            'is_print_compare_log': True
        }
        with open(self.output_json[2], 'w') as f:
            json.dump(input_param2, f, indent=4)
        args = Args(input_path=self.output_json[2], output_path=self.output, framework='pytorch')
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, 'Successfully exported compare graph results.')

        input_param3 = {
            'npu_path': self.input,
            'bench_path': self.input,
            'is_print_compare_log': True
        }
        with open(self.output_json[3], 'w') as f:
            json.dump(input_param3, f, indent=4)
        args = Args(input_path=self.output_json[3], output_path=self.output, framework='pytorch')
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, 'Successfully exported compare graph results.')

        input_param4 = {
            'npu_path': os.path.join(self.input, 'step0'),
            'is_print_compare_log': True
        }
        with open(self.output_json[4], 'w') as f:
            json.dump(input_param4, f, indent=4)
        args = Args(input_path=self.output_json[4], output_path=self.output, framework='pytorch')
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, "Successfully exported build graph results.")

        input_param5 = {
            'npu_path': self.input,
            'is_print_compare_log': True
        }
        with open(self.output_json[5], 'w') as f:
            json.dump(input_param5, f, indent=4)
        args = Args(input_path=self.output_json[5], output_path=self.output, framework='pytorch')
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, "Successfully exported build graph results.")

        input_param6 = {
            'npu_path': self.input,
            'bench_path': os.path.join(self.input, 'step0'),
            'is_print_compare_log': True
        }
        with open(self.output_json[6], 'w') as f:
            json.dump(input_param6, f, indent=4)
        args = Args(input_path=self.output_json[6], output_path=self.output, framework='pytorch')
        with self.assertRaises(ValueError):
            _graph_service_command(args)

    def test_graph_service_parser(self):
        parser = argparse.ArgumentParser()
        _graph_service_parser(parser)
        args = parser.parse_args(['-i', 'input.json', '-o', 'output.json'])
        self.assertEqual(args.input_path, 'input.json')
        self.assertEqual(args.output_path, 'output.json')

        args = parser.parse_args(['-i', 'input.json', '-o', 'output.json', '-lm', 'mapping.json'])
        self.assertEqual(args.layer_mapping, 'mapping.json')

        args = parser.parse_args(['-i', 'input.json', '-o', 'output.json', '-oc'])
        self.assertTrue(args.overflow_check)

        args = parser.parse_args(['-i', 'input.json', '-o', 'output.json'])
        self.assertFalse(args.overflow_check)

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)
        for json_data in self.output_json:
            if os.path.exists(json_data):
                os.remove(json_data)


if __name__ == '__main__':
    unittest.main()
