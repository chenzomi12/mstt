import csv
import os
import shutil
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch.nn.functional

from msprobe.core.common.utils import CompareException
from msprobe.pytorch.api_accuracy_checker.compare.compare import Comparator
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import DETAIL_TEST_ROWS
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import CompareColumn
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import UtDataInfo

current_time = time.strftime("%Y%m%d%H%M%S")
RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + '.csv'
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.output_path = os.path.join(base_dir, "../compare_result")
        os.mkdir(self.output_path, mode=0o750)
        self.result_csv_path = os.path.join(self.output_path, RESULT_FILE_NAME)
        self.details_csv_path = os.path.join(self.output_path, DETAILS_FILE_NAME)
        self.is_continue_run_ut = False
        self.compare = Comparator(self.result_csv_path, self.details_csv_path, self.is_continue_run_ut)

    def tearDown(self) -> None:
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

    def test_compare_dropout_pass_large_tensor(self):
        # Arrange
        bench_output = torch.tensor([0, 0, 1, 1, 0, 1] * 20)  # 120 elements, 60 zeros
        device_output = torch.tensor([0, 0, 1, 1, 0, 1] * 20)  # Same as bench_output
        # Act
        result = self.compare._compare_dropout(bench_output, device_output)
        # Assert
        self.assertEqual(result, ('pass', 1))

    def test_compare_dropout_error_large_tensor(self):
        # Arrange
        bench_output = torch.tensor([0, 0, 1, 1, 0, 1] * 20)  # 120 elements, 60 zeros
        device_output = torch.tensor([1, 1, 1, 1, 1, 1] * 20)  # 120 elements, 0 zeros
        # Act
        result = self.compare._compare_dropout(bench_output, device_output)
        # Assert
        self.assertEqual(result, ('error', 0))

    def test_compare_dropout_pass_small_tensor(self):
        # Arrange
        bench_output = torch.tensor([0, 1, 0])  # 3 elements
        device_output = torch.tensor([0, 1, 0])  # Same as bench_output
        # Act
        result = self.compare._compare_dropout(bench_output, device_output)
        # Assert
        self.assertEqual(result, ('pass', 1))

    def test_compare_dropout_large_tensor_boundary(self):
        # Arrange
        bench_output = torch.tensor([0, 0, 1] * 33 + [0])  # 100 elements, 67 zeros
        device_output = torch.tensor([0, 1, 1] * 33 + [0])  # 100 elements, 34 zeros
        # Act
        result = self.compare._compare_dropout(bench_output, device_output)
        # Assert
        self.assertEqual(result, ('error', 0))

    def test_compare_core_wrapper(self):
        dummy_input = torch.randn(100, 100)
        bench_out, npu_out = dummy_input, dummy_input
        test_final_success, detailed_result_total = self.compare._compare_core_wrapper("api", bench_out, npu_out)
        actual_cosine_similarity = detailed_result_total[0][3]
        # 设置一个小的公差值
        tolerance = 1e-4
        # 判断实际的余弦相似度值是否在预期值的公差范围内
        self.assertTrue(np.isclose(actual_cosine_similarity, 1.0, atol=tolerance))
        # 对其他值进行比较，确保它们符合预期
        detailed_result_total[0][3] = 1.0
        self.assertEqual(detailed_result_total, [['torch.float32', 'torch.float32', (100, 100), 1.0, 0.0, ' ', ' ', ' ',
                                                ' ', 0.0, 0.0, 0, 0.0, 0.0, ' ', ' ', ' ', ' ', ' ', ' ', 'pass', 
                                                '\nMax abs error is less than 0.001, consider as pass, skip other check and set to SPACE.\n']])
        self.assertTrue(test_final_success)

        bench_out, npu_out = [dummy_input, dummy_input], [dummy_input, dummy_input]
        test_final_success, detailed_result_total = self.compare._compare_core_wrapper("api", bench_out, npu_out)
        actual_cosine_similarity = detailed_result_total[0][3]
        self.assertTrue(np.isclose(actual_cosine_similarity, 1.0, atol=tolerance))
        actual_cosine_similarity = detailed_result_total[1][3]
        self.assertTrue(np.isclose(actual_cosine_similarity, 1.0, atol=tolerance))
        detailed_result_total[0][3] = 1.0
        detailed_result_total[1][3] = 1.0
        self.assertTrue(test_final_success)
        self.assertEqual(detailed_result_total, [['torch.float32', 'torch.float32', (100, 100), 1.0, 0.0, ' ', ' ', ' ',
                                                ' ', 0.0, 0.0, 0, 0.0, 0.0, ' ', ' ', ' ', ' ', ' ', ' ', 'pass', 
                                                '\nMax abs error is less than 0.001, consider as pass, skip other check and set to SPACE.\n'], 
                                                 ['torch.float32', 'torch.float32', (100, 100), 1.0, 0.0, ' ', ' ', ' ', 
                                                ' ', 0.0, 0.0, 0, 0.0, 0.0, ' ', ' ', ' ', ' ', ' ', ' ', 'pass', 
                                                '\nMax abs error is less than 0.001, consider as pass, skip other check and set to SPACE.\n']])

    def test_compare_core_different(self):
        res = self.compare._compare_core('api', 1, 'str')

        self.assertEqual(res[0], 'error')
        self.assertEqual(res[2], 'bench and npu output type is different.')

    def test_compare_core_with_dict(self):
        output_dict = {
            'key1': 1,
            'key2': 2
        }
        res = self.compare._compare_core('api', output_dict, output_dict)
        
        self.assertEqual(res[0], 'error')
        self.assertEqual(res[2], "Unexpected output type in compare_core: <class 'list'>")
    
    def test_compare_core_with_dict_different(self):
        bench_dict = {
            'key1': 1,
            'key2': 2
        }
        device_dict = {
            'key3': 3,
            'key4': 4
        }
        res = self.compare._compare_core('api', bench_dict, device_dict)

        self.assertEqual(res[0], 'error')
        self.assertEqual(res[2], 'bench and npu output dict keys are different.')

    def test_compare_core_with_tensor(self):
        tensor = torch.tensor([1, 2, 3])
        res = self.compare._compare_core('api', tensor, tensor)

        self.assertEqual(res[0], 'pass')
        self.assertEqual(res[2], 'Compare algorithm is not supported for int64 data. Only judged by Error Rate.\n')

    def test_compare_core_with_buildin(self):
        interger = 1
        res = self.compare._compare_core('api', interger, interger)

        self.assertEqual(res[0], 'pass')
        self.assertEqual(res[2], '')

    def test_compare_core_with_none(self):
        res = self.compare._compare_core('api', None, None)

        self.assertEqual(res[0], 'SKIP')
        self.assertEqual(res[2], 'Bench output is None, skip this test.')

    def test_compare_output(self):
        bench_out, npu_out = torch.randn(100, 100), torch.randn(100, 100)
        bench_grad, npu_grad = [torch.randn(100, 100)], [torch.randn(100, 100)]
        api_name = 'Functional.conv2d.0'
        data_info = UtDataInfo(bench_grad, npu_grad, bench_out, npu_out, None, None, None)
        is_fwd_success, is_bwd_success = self.compare.compare_output(api_name, data_info)
        self.assertFalse(is_fwd_success)
        # is_bwd_success should be checked

        dummy_input = torch.randn(100, 100)
        bench_out, npu_out = dummy_input, dummy_input
        data_info = UtDataInfo(None, None, bench_out, npu_out, None, None, None)
        is_fwd_success, is_bwd_success = self.compare.compare_output(api_name, data_info)
        self.assertTrue(is_fwd_success)
        self.assertTrue(is_bwd_success)

    def test_compare_output_error(self):
        bench_out, npu_out = torch.randn(100, 100), torch.randn(100, 100)
        bench_grad, npu_grad = [torch.randn(100, 100)], [torch.randn(100, 100)]
        api_name = 'Functional.conv2d'
        data_info = UtDataInfo(bench_grad, npu_grad, bench_out, npu_out, None, None, None)

        with self.assertRaises(ValueError):
            self.compare.compare_output(api_name, data_info)

    def test_compare_output_with_dropout(self):
        bench_out, npu_out = torch.randn(100, 100), torch.randn(100, 100)
        bench_grad, npu_grad = [torch.randn(100, 100)], [torch.randn(100, 100)]
        api_name = 'Functional.dropout.0'
        data_info = UtDataInfo(bench_grad, npu_grad, bench_out, npu_out, None, None, None)
        is_fwd_success, is_bwd_success = self.compare.compare_output(api_name, data_info)
        self.assertTrue(is_fwd_success)
        self.assertTrue(is_bwd_success)

    def test_compare_output_with_backward_message(self):
        bench_out, npu_out = torch.randn(100, 100), torch.randn(100, 100)
        bench_grad, npu_grad = [torch.randn(100, 100)], [torch.randn(100, 100)]
        api_name = 'Functional.conv2d.0'
        data_info = UtDataInfo(bench_grad, npu_grad, bench_out, npu_out, None, "test_message", None)
        is_fwd_success, is_bwd_success = self.compare.compare_output(api_name, data_info)
        self.assertFalse(is_fwd_success)
        self.assertFalse(is_fwd_success)

    def test_record_results(self):
        args = ('Functional.conv2d.0', False, 'N/A', [['torch.float64', 'torch.float32', (32, 64, 112, 112), 1.0,
                                                       0.012798667686, 'N/A', 0.81631212311, 0.159979121213, 'N/A',
                                                       'error', '\n']], None, 0)
        self.compare.record_results(args)
        with open(self.details_csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            api_name_list = [row[0] for row in csv_reader]
        self.assertEqual(api_name_list[0], 'Functional.conv2d.0.forward.output.0')

    def test_compare_torch_tensor(self):
        cpu_output = torch.Tensor([1.0, 2.0, 3.0])
        npu_output = torch.Tensor([1.0, 2.0, 3.0])
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_torch_tensor("api", cpu_output, npu_output,
                                                                             compare_column)
        self.assertEqual(status, "pass")

    def test_compare_torch_tensor_bf16(self):
        cpu_output = torch.Tensor([1.0, 2.0, 3.0])
        npu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_torch_tensor("api", cpu_output, npu_output,
                                                                             compare_column)
        self.assertEqual(status, "pass")

    def test_compare_torch_tensor_different_shape(self):
        cpu_output = torch.Tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        npu_output = torch.Tensor([1.0, 2.0, 3.0])
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_torch_tensor("api", cpu_output, npu_output,
                                                                             compare_column)
        self.assertEqual(status, "error")

    def test_compare_torch_tensor_different_dtype(self):
        cpu_output = torch.Tensor([True, True, False])
        npu_output = torch.Tensor([1.0, 2.0, 3.0])
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_torch_tensor("api", cpu_output, npu_output,
                                                                             compare_column)
        self.assertEqual(status, "error")

    def test_compare_torch_tensor_special_dtype(self):
        cpu_output = torch.Tensor([True, True, False])
        npu_output = torch.Tensor([True, True, False])
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_torch_tensor("api", cpu_output, npu_output,
                                                                             compare_column)
        self.assertEqual(status, "pass")

    def test_compare_builtin_type_pass_with_special_types(self):
        compare_column = CompareColumn()
        bench_out = 1
        npu_out = 1
        status, compare_result, message = self.compare._compare_builtin_type(bench_out, npu_out, compare_column)
        self.assertEqual((status, compare_result.error_rate, message), ('pass', 0, ''))

    def test_compare_builtin_type_pass_with_none_special_types(self):
        compare_column = CompareColumn()
        bench_out = np.array([1])
        npu_out = np.array([1])
        status, compare_result, message = self.compare._compare_builtin_type(bench_out, npu_out, compare_column)
        self.assertEqual((status, compare_result.error_rate, message), ('pass', ' ', ''))

    def test_compare_builtin_type_error(self):
        compare_column = CompareColumn()
        bench_out = 1
        npu_out = 2
        status, compare_result, message = self.compare._compare_builtin_type(bench_out, npu_out, compare_column)
        self.assertEqual((status, compare_result.error_rate, message), ('error', ' ', ''))

    def test_compare_float_tensor(self):
        cpu_output = torch.Tensor([1.0, 2.0, 3.0])
        npu_output = torch.Tensor([1.0, 2.0, 3.0])
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("conv2d", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)
        self.assertEqual(status, "pass")

    def test_compare_float_tensor_binary(self):
        cpu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        npu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("abs", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)
        self.assertEqual(status, "pass")

    def test_compare_float_tensor_absolute(self):
        cpu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        npu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("mul", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)

        self.assertEqual(status, "pass")

    def test_compare_float_tensor_ulp(self):
        cpu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        npu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("__matmul__", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)

        self.assertEqual(status, "pass")

    def test_compare_float_tensor_error_16(self):
        cpu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        npu_output = torch.tensor([1.1, 2.1, 3.1], dtype=torch.float16)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("__matmul__", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)

        self.assertEqual(status, "error")

    def test_compare_float_tensor_pass_16(self):
        cpu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        npu_output = torch.tensor([1.0001, 2.0001, 3.0001], dtype=torch.float16)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("__matmul__", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)

        self.assertEqual(status, "pass")

    def test_compare_float_tensor_warn_16(self):
        cpu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        npu_output = torch.tensor([1.01, 2.01, 3.01], dtype=torch.float16)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("__matmul__", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)

        self.assertEqual(status, "Warning")

    def test_compare_float_tensor_error_32(self):
        cpu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        npu_output = torch.tensor([1.01, 2.01, 3.01], dtype=torch.float32)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("__matmul__", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)

        self.assertEqual(status, "error")

    def test_compare_float_tensor_pass_32(self):
        cpu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        npu_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        compare_column = CompareColumn()
        status, compare_column, message = self.compare._compare_float_tensor("__matmul__", cpu_output.numpy(),
                                                                             npu_output.numpy(),
                                                                             compare_column, npu_output.dtype)

        self.assertEqual(status, "pass")

    def test_get_run_ut_detail_success(self):
        # Arrange
        test_result = [
            "test_subject",  # subject_prefix
            None,            # Placeholder for other indices
            None,
            [[0.123456, 1.234567], [2.345678]],  # fwd_result
            [[3.456789], [4.567890]]              # bwd_result
        ]
        
        # Act
        result = self.compare._get_run_ut_detail(test_result)

        # Assert
        expected_result = [
            ["test_subject.forward.output.0", "0.12345600000000", "1.23456700000000"],
            ["test_subject.forward.output.1", "2.34567800000000"],
            ["test_subject.backward.output.0", "3.45678900000000"],
            ["test_subject.backward.output.1", "4.56789000000000"],
        ]
        self.assertEqual(result, expected_result)

    @patch('msprobe.pytorch.api_accuracy_checker.compare.compare.logger')
    def test_get_run_ut_detail_index_error(self, mock_logger):
        # Arrange
        test_result = [
            "test_subject",  # subject_prefix
            None         # Placeholder for other indices
        ]

        # Act and Assert
        with self.assertRaises(CompareException):
            self.compare._get_run_ut_detail(test_result)
        mock_logger.error.assert_called_once_with("List index out of bounds when writing detail CSV.")

    @patch('msprobe.pytorch.api_accuracy_checker.compare.compare.write_csv')
    @patch('msprobe.pytorch.api_accuracy_checker.compare.compare.os.path.exists')
    def test_write_csv_title(self, mock_exists, mock_write_csv):
        # Mock the behavior of os.path.exists
        mock_exists.return_value = False

        self.compare.save_path_list = ['result.csv', 'summary.csv']
        self.compare.detail_save_path_list = ['detail_result.csv', 'detail_summary.csv']

        # Act
        self.compare.write_csv_title()

        # Assert
        # Check that write_csv was called for the paths that do not exist
        mock_write_csv.assert_any_call(
            [[self.compare.COLUMN_API_NAME, 
              self.compare.COLUMN_FORWARD_SUCCESS,
              self.compare.COLUMN_BACKWARD_SUCCESS,
              "Message"]],
            'summary.csv'
        )
        mock_write_csv.assert_any_call(DETAIL_TEST_ROWS, 'detail_result.csv')

        # Ensure write_csv was not called for 'result.csv' and 'detail_summary.csv'
        self.assertEqual(mock_write_csv.call_count, 4)

    @patch('msprobe.pytorch.api_accuracy_checker.compare.compare.write_csv')
    @patch('msprobe.pytorch.api_accuracy_checker.compare.compare.logger')
    def test_write_summary_csv(self, mock_logger, mock_write_csv):
        self.compare.stack_info = {'test1': ['info1', 'info2']}
        self.compare.save_path_list = ['summary_result.csv', 'summary_detail.csv']
        test_result = [
            'test1',  # name
            'SKIP',  # status
            None,  # Placeholder
            [[{'message': 'Skipped test'}]],  # Group message
            0  # rank
        ]

        # Act
        self.compare.write_summary_csv(test_result)

        # Assert
        mock_write_csv.assert_called_once()

    @patch('msprobe.pytorch.api_accuracy_checker.compare.compare.write_csv')
    @patch('msprobe.pytorch.api_accuracy_checker.compare.compare.Comparator._get_run_ut_detail')
    @patch('msprobe.pytorch.api_accuracy_checker.compare.compare.Comparator.get_path_from_rank')
    def test_write_detail_csv(self, mock_get_path, mock_get_detail, mock_write_csv):
        test_result = ['test_result']
        self.compare.write_detail_csv(test_result)
        
        mock_get_detail.assert_called_once()
        mock_get_path.assert_called_once()
        mock_write_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()
