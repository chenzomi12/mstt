import unittest
import torch

from msprobe.pytorch.bench_functions.linear import npu_linear, npu_linear_backward


class TestNPULinear(unittest.TestCase):

    def setUp(self):
        self.x = torch.randn(2, 3)  # 输入数据
        self.weight = torch.randn(4, 3)  # 权重
        self.bias = torch.randn(4)  # 偏置
        self.grad = torch.randn(2, 4)  # 梯度

    def test_npu_linear_with_bias(self):
        result = npu_linear(self.x, self.weight, self.bias)
        expected_result = torch.nn.functional.linear(self.x, self.weight, self.bias)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_npu_linear_without_bias(self):
        result = npu_linear(self.x, self.weight, None)
        expected_result = torch.nn.functional.linear(self.x, self.weight, None)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_npu_linear_backward(self):
        input_grad, weight_grad = npu_linear_backward(self.grad, self.x, self.weight)

        # 计算预期的输入梯度
        expected_input_grad = torch.matmul(self.grad, self.weight)
        self.assertTrue(torch.allclose(input_grad, expected_input_grad, atol=1e-6))

        # 计算预期的权重梯度
        expected_weight_grad = torch.matmul(self.grad.t(), self.x)
        self.assertTrue(torch.allclose(weight_grad, expected_weight_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
