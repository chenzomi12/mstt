import unittest
from unittest.mock import Mock
import numpy as np

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_register import StandardRegistry
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import (
    absolute_standard_api, binary_standard_api, BINARY_COMPARE_UNSUPPORT_LIST
)

class TestStandardRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = StandardRegistry()
        
    def test_register_valid_function(self):
        """测试正常注册比较函数"""
        mock_func = Mock()
        self.registry.register("test_standard", mock_func)
        self.assertEqual(self.registry.comparison_functions["test_standard"], mock_func)

    def test_register_invalid_function(self):
        """测试注册非callable对象时抛出异常"""
        with self.assertRaises(ValueError):
            self.registry.register("test_standard", "not_callable")

    def test_get_comparison_function_binary_consistency(self):
        """测试获取二进制一致性比较函数"""
        mock_func = Mock()
        self.registry.register("binary_consistency", mock_func)
        # 使用支持二进制比较的数据类型
        result = self.registry.get_comparison_function("abs", dtype='torch.int8')
        self.assertEqual(result, mock_func)

    def test_get_comparison_function_absolute_threshold(self):
        """测试获取绝对阈值比较函数"""
        mock_func = Mock()
        self.registry.register("absolute_threshold", mock_func)
        # 假设'test_api'在absolute_standard_api列表中
        result = self.registry.get_comparison_function("mul")
        self.assertEqual(result, mock_func)

    def test_get_comparison_function_ulp(self):
        """测试获取ULP比较函数"""
        mock_func = Mock()
        self.registry.register("ulp_compare", mock_func)
        result = self.registry.get_comparison_function("matmul")
        self.assertEqual(result, mock_func)

    def test_get_comparison_function_thousandth(self):
        """测试获取双千比较函数"""
        mock_func = Mock()
        self.registry.register("thousandth_threshold", mock_func)
        result = self.registry.get_comparison_function("conv2d")
        self.assertEqual(result, mock_func)

    def test_get_comparison_function_benchmark(self):
        """测试获取默认benchmark比较函数"""
        mock_func = Mock()
        self.registry.register("benchmark", mock_func)
        result = self.registry.get_comparison_function("npu_fusion_attention")
        self.assertEqual(result, mock_func)

    def test_get_standard_category_binary(self):
        """测试获取二进制一致性标准类别"""
        dtype = 'torch.int8'
        self.assertNotIn(dtype, BINARY_COMPARE_UNSUPPORT_LIST)
        category = self.registry._get_standard_category("abs", dtype)
        self.assertEqual(category, "binary_consistency")

    def test_get_standard_category_absolute(self):
        """测试获取绝对阈值标准类别"""
        category = self.registry._get_standard_category("mul")
        self.assertEqual(category, "absolute_threshold")

    def test_get_standard_category_default(self):
        """测试获取默认benchmark标准类别"""
        category = self.registry._get_standard_category("unknown_api")
        self.assertEqual(category, "benchmark")

    def test_get_standard_category_ulp(self):
        """测试获取ULP标准类别"""
        category = self.registry._get_standard_category("matmul")
        self.assertEqual(category, "ulp_compare")

    def test_get_standard_category_thousandth(self):
        """测试获取双千比对标准类别"""
        category = self.registry._get_standard_category("conv2d")
        self.assertEqual(category, "thousandth_threshold")


if __name__ == '__main__':
    unittest.main()