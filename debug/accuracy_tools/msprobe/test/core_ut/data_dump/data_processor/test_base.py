import unittest
from unittest.mock import patch, MagicMock
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from msprobe.core.common.log import logger
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs, \
    TensorStatInfo, BaseDataProcessor
from msprobe.core.data_dump.data_processor.mindspore_processor import MindsporeDataProcessor


class TestModuleForwardInputsOutputs(unittest.TestCase):

    @patch('msprobe.core.common.utils.convert_tuple')
    def test_args_tuple(self, mock_convert_tuple):
        mock_convert_tuple.return_value = (1, 2, 3)
        module = ModuleForwardInputsOutputs(args=(1, 2, 3), kwargs=None, output=None)
        self.assertEqual(module.args_tuple, (1, 2, 3))

    @patch('msprobe.core.common.utils.convert_tuple')
    def test_output_tuple(self, mock_convert_tuple):
        mock_convert_tuple.return_value = (4, 5, 6)
        module = ModuleForwardInputsOutputs(args=None, kwargs=None, output=(4, 5, 6))
        self.assertEqual(module.output_tuple, (4, 5, 6))

    def test_update_output_with_args_and_kwargs(self):
        module = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3, 'b': 4}, output=None)
        module.update_output_with_args_and_kwargs()
        self.assertEqual(module.output, (1, 2, 3, 4))


class TestModuleBackwardInputsOutputs(unittest.TestCase):

    @patch('msprobe.core.common.utils.convert_tuple')
    def test_grad_input_tuple(self, mock_convert_tuple):
        mock_convert_tuple.return_value = (1, 2, 3)
        module = ModuleBackwardInputsOutputs(grad_output=None, grad_input=(1, 2, 3))
        self.assertEqual(module.grad_input_tuple, (1, 2, 3))

    @patch('msprobe.core.common.utils.convert_tuple')
    def test_grad_output_tuple(self, mock_convert_tuple):
        mock_convert_tuple.return_value = (4, 5, 6)
        module = ModuleBackwardInputsOutputs(grad_output=(4, 5, 6), grad_input=None)
        self.assertEqual(module.grad_output_tuple, (4, 5, 6))


class TestTensorStatInfo(unittest.TestCase):

    def test_tensor_stat_info(self):
        tensor_info = TensorStatInfo(max_val=10, min_val=1, mean_val=5, norm_val=3)
        self.assertEqual(tensor_info.max, 10)
        self.assertEqual(tensor_info.min, 1)
        self.assertEqual(tensor_info.mean, 5)
        self.assertEqual(tensor_info.norm, 3)


class TestBaseDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = BaseDataProcessor(self.config, self.data_writer)
        self.data_writer.dump_tensor_data_dir = "./dump_data"
        self.processor.current_api_or_module_name = "test_api"
        self.processor.api_data_category = "input"

    @patch('inspect.stack')
    def test_analyze_api_call_stack(self, mock_stack):
        mock_stack.return_value = [
            (None, 'file3.py', 30, 'function3', ['code line 3'], None),
            (None, 'file1.py', 40, 'function1', ['code line 1'], None),
            (None, 'file2.py', 50, 'function2', ['code line 2'], None),
            (None, 'file3.py', 60, 'function3', ['code line 3'], None),
            (None, 'file1.py', 70, 'function1', ['code line 1'], None),
            (None, 'file1.py', 80, 'function1', ['code line 1'], None),
            (None, 'file2.py', 90, 'function2', ['code line 2'], None),
            (None, 'file3.py', 100, 'function3', ['code line 3'], None)
        ]
        result = BaseDataProcessor.analyze_api_call_stack('test_stack')
        expected_output = (
            'File file1.py, line 80, in function1, \n code line 1',
            'File file2.py, line 90, in function2, \n code line 2',
            'File file3.py, line 100, in function3, \n code line 3',
        )

        self.assertEqual(result, expected_output)

    def test_analyze_builtin(self):
        result = self.processor._analyze_builtin(slice(1, 10, 2))
        expected = {'type': 'slice', 'value': [1, 10, 2]}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(slice(1, np.int64(10), np.int64(2)))
        expected = {'type': 'slice', 'value': [1, 10, 2]}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(...)
        expected = {'type': 'ellipsis', 'value': "..."}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(1)
        expected = {'type': 'int', 'value': 1}
        self.assertEqual(result, expected)

    def test_analyze_numpy(self):
        result = BaseDataProcessor._analyze_numpy(np.int32(5))
        expected = {"type": 'int32', "value": 5}
        self.assertEqual(result, expected)

        result = BaseDataProcessor._analyze_numpy(np.float32(3.14))
        expected = {"type": 'float32', "value": 3.140000104904175}
        self.assertEqual(result, expected)

        result = BaseDataProcessor._analyze_numpy(np.bool_(True))
        expected = {"type": 'bool_', "value": True}
        self.assertEqual(result, expected)

        result = BaseDataProcessor._analyze_numpy(np.str_("abc"))
        expected = {"type": 'str_', "value": "abc"}
        self.assertEqual(result, expected)

        result = BaseDataProcessor._analyze_numpy(np.byte(1))
        expected = {"type": 'int8', "value": 1}
        self.assertEqual(result, expected)

        result = BaseDataProcessor._analyze_numpy(np.complex128(1 + 2j))
        expected = {"type": 'complex128', "value": (1 + 2j)}
        self.assertEqual(result, expected)

    def test_get_special_types(self):
        self.assertIn(int, BaseDataProcessor.get_special_types())

    def test_analyze_ndarray(self):
        ndarray = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        result = BaseDataProcessor._analyze_ndarray(ndarray, 'numpy.ndarray')
        expected_result = {
            'type': 'numpy.ndarray',
            'dtype': 'int32',
            'shape': (2, 3),
            'Max': 6,
            'Min': 1,
            'Mean': 3.5,
            'Norm': 9.539392014169456
        }
        self.assertEqual(result, expected_result)

        ndarray = np.array([], dtype=np.int32)
        result = BaseDataProcessor._analyze_ndarray(ndarray, 'numpy.ndarray')
        expected_result = {
            'type': 'numpy.ndarray',
            'dtype': 'int32',
            'shape': (0,),
            'Max': None,
            'Min': None,
            'Mean': None,
            'Norm': None
        }
        self.assertEqual(result, expected_result)

    def test_recursive_apply_transform(self):
        transform = lambda x, _: x * 2
        Test = namedtuple("Test", ['a'])
        myNamedTuple = Test(1)

        @dataclass
        class MyDataClass:
            last_hidden_state: int = None
            hidden_states: Optional[Tuple[int, ...]] = None
            attentions: Optional[Tuple[int, ...]] = None

        myData = MyDataClass(
            last_hidden_state=1,
            hidden_states=(2, 3),
            attentions=(4, 5)
        )
        expected_dataclass_res = {'last_hidden_state': 2, 'hidden_states': [4, 6], 'attentions': [8, 10]}
        self.assertEqual(BaseDataProcessor.recursive_apply_transform(2, transform), 4)
        self.assertEqual(BaseDataProcessor.recursive_apply_transform(myData, transform), expected_dataclass_res)
        self.assertEqual(BaseDataProcessor.recursive_apply_transform(myNamedTuple, transform), {'a': 2})
        self.assertEqual(BaseDataProcessor.recursive_apply_transform([1, 2], transform), [2, 4])
        self.assertEqual(BaseDataProcessor.recursive_apply_transform((1, 2), transform), [2, 4])
        self.assertEqual(BaseDataProcessor.recursive_apply_transform({'a': 1}, transform), {'a': 2})

    @patch.object(logger, 'debug')
    def test_recursive_apply_transform_with_warning(self, mock_logger):
        transform = lambda x, _: x * 2
        BaseDataProcessor.recursive_apply_transform({1, 2, 3}, transform)
        mock_logger.assert_called_with(f"Data type {type({1, 2, 3})} is not supported.")

    def test_if_return_forward_new_output(self):
        self.processor._return_forward_new_output = True
        self.assertTrue(self.processor.if_return_forward_new_output())

    def test_get_forward_new_output(self):
        self.processor._return_forward_new_output = True
        self.processor._forward_new_output = "new_output"
        self.assertEqual(self.processor.get_forward_new_output(), "new_output")
        self.assertFalse(self.processor._return_forward_new_output)

    def test_update_iter(self):
        self.processor.update_iter(5)
        self.assertEqual(self.processor.current_iter, 5)

    def test_update_api_or_module_name(self):
        self.processor.update_api_or_module_name("new_api")
        self.assertEqual(self.processor.current_api_or_module_name, "new_api")

    def test_is_dump_for_data_mode(self):
        self.config.data_mode = ["all"]
        self.processor.allowed_data_mode = self.processor._get_allowed_data_mode(self.config.data_mode)
        self.assertTrue(self.processor.is_dump_for_data_mode("forward", "input"))

        self.config.data_mode = ["forward"]
        self.processor.allowed_data_mode = self.processor._get_allowed_data_mode(self.config.data_mode)
        self.assertTrue(self.processor.is_dump_for_data_mode("forward", "input"))

        self.config.data_mode = ["input"]
        self.processor.allowed_data_mode = self.processor._get_allowed_data_mode(self.config.data_mode)
        self.assertTrue(self.processor.is_dump_for_data_mode("forward", "input"))

        self.config.data_mode = ["backward"]
        self.processor.allowed_data_mode = self.processor._get_allowed_data_mode(self.config.data_mode)
        self.assertFalse(self.processor.is_dump_for_data_mode("forward", "input"))

        self.config.data_mode = ["forward", "input"]
        self.processor.allowed_data_mode = self.processor._get_allowed_data_mode(self.config.data_mode)
        self.assertFalse(self.processor.is_dump_for_data_mode("forward", "output"))

        self.config.data_mode = ["forward", "input"]
        self.processor.allowed_data_mode = self.processor._get_allowed_data_mode(self.config.data_mode)
        self.assertFalse(self.processor.is_dump_for_data_mode("backward", "input"))

    @patch.object(BaseDataProcessor, 'analyze_element')
    def test_analyze_forward_input(self, mock_analyze_element):
        mock_analyze_element.side_effect = lambda args: args
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=None)
        self.config.data_mode = ["all"]
        result = self.processor.analyze_forward_input("test_forward_input", None, module_io)
        expected = {
            "test_forward_input": {
                "input_args": (1, 2),
                "input_kwargs": {'a': 3}
            }
        }
        self.assertEqual(result, expected)

    @patch.object(BaseDataProcessor, 'analyze_element')
    def test_analyze_forward_output(self, mock_analyze_element):
        mock_analyze_element.side_effect = lambda args: args
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=(4, 5))
        self.config.data_mode = ["all"]
        result = self.processor.analyze_forward_output("test_forward_output", None, module_io)
        expected = {
            "test_forward_output": {
                "output": (4, 5)
            }
        }
        self.assertEqual(result, expected)

    @patch.object(BaseDataProcessor, 'analyze_element')
    def test_analyze_forward(self, mock_analyze_element):
        mock_analyze_element.side_effect = lambda args: args
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=(4, 5))
        self.config.data_mode = ["all"]
        result = self.processor.analyze_forward("test_forward", None, module_io)
        expected = {
            "test_forward": {
                "input_args": (1, 2),
                "input_kwargs": {'a': 3},
                "output": (4, 5)
            }
        }
        self.assertEqual(result, expected)

    @patch.object(BaseDataProcessor, 'analyze_element')
    def test_analyze_backward(self, mock_analyze_element):
        mock_analyze_element.side_effect = lambda args: args
        module_io = ModuleBackwardInputsOutputs(grad_input=(1, 2), grad_output=(3, 4))
        self.config.data_mode = ["all"]
        result = self.processor.analyze_backward("test_backward", None, module_io)
        expected = {
            "test_backward": {
                "input": (1, 2),
                "output": (3, 4)
            }
        }
        self.assertEqual(result, expected)

    def test_get_save_file_path(self):
        self.config.framework = "pytorch"
        result = self.processor.get_save_file_path("suffix")
        expected_file_name = "test_api.input.suffix.pt"
        expected_file_path = os.path.join(self.data_writer.dump_tensor_data_dir, expected_file_name)
        self.assertEqual(result, (expected_file_name, expected_file_path))

    def test_get_save_file_path_with_save_name(self):
        self.config.framework = "pytorch"
        self.processor.save_name = "custom_name"
        result = self.processor.get_save_file_path("suffix")
        expected_file_name = "custom_name.pt"
        expected_file_path = os.path.join(self.data_writer.dump_tensor_data_dir, expected_file_name)
        self.assertEqual(result, (expected_file_name, expected_file_path))

    def test_set_value_into_nested_structure(self):
        dst_data_structure = {"key1": [None, None]}
        self.processor.set_value_into_nested_structure(dst_data_structure, ["key1", 0], 12)
        excepted_result = {"key1": [12, None]}
        self.assertEqual(dst_data_structure, excepted_result)

    def test_analyze_element_to_all_none(self):
        element = {"key1": [12, 3, {"key2": 10, "key3": ["12"]}]}
        result = self.processor.analyze_element_to_all_none(element)
        excepted_result = {"key1": [None, None, {"key2": None, "key3": [None]}]}
        self.assertEqual(result, excepted_result)

    @patch.object(MindsporeDataProcessor, "is_hookable_element", return_value=True)
    def test_register_hook_single_element(self, _):
        element = MagicMock()
        element.hasattr = MagicMock(side_effect=lambda attr: attr == "register_hook")
        element.requires_grad = True
        hook_fn = MagicMock()
        MindsporeDataProcessor.register_hook_single_element(element, [1, 2], hook_fn)
        element.register_hook.assert_called_once()

    @patch("msprobe.core.data_dump.data_processor.base.partial")
    def test_analyze_debug_backward(self, mock_partial):
        variable = MagicMock()  # 模拟输入变量
        grad_name_with_count = "grad_name_1"
        nested_data_structure = {"key": "value"}  # 模拟嵌套数据结构

        self.processor.recursive_apply_transform = MagicMock()
        self.processor.set_value_into_nested_structure = MagicMock()
        self.processor.analyze_element = MagicMock(return_value="grad_data_info")
        self.processor.register_hook_single_element = MagicMock()

        # call
        self.processor.analyze_debug_backward(variable, grad_name_with_count, nested_data_structure)

        # check partial
        args, kwargs = mock_partial.call_args
        self.assertIn("hook_fn", kwargs)
        self.assertEqual(args[0], self.processor.register_hook_single_element)
        self.assertEqual(kwargs["hook_fn"].__name__, "hook_fn")

        wrap_func = mock_partial.return_value
        self.processor.recursive_apply_transform.assert_called_once_with(variable, wrap_func)

        grad = MagicMock()
        index = ["layer1", "layer2"]
        result = kwargs["hook_fn"](grad, index)

        # 验证 hook_fn 内部逻辑
        self.processor.analyze_element.assert_called_once_with(grad)
        self.processor.set_value_into_nested_structure.assert_called_once_with(
            nested_data_structure, ["grad_name_1", "layer1", "layer2"], "grad_data_info"
        )
        self.assertIsNone(self.processor.save_name)
        self.assertEqual(result, grad)
