import unittest
import torch

from msprobe.pytorch.bench_functions.confusion_transpose import npu_confusion_transpose, \
    npu_confusion_transpose_backward


class TestNPUConfusionTranspose(unittest.TestCase):
    def setUp(self):
        self.data = torch.arange(24).reshape(2, 3, 4)
        self.perm = (2, 0, 1)  # 置换
        self.shape = (4, 2, 3)  # 形状
        self.transpose_first = True  # 是否先进行转置

    def test_transpose_first_true(self):
        output = npu_confusion_transpose(self.data, self.perm, self.shape, True)
        expected_output = self.data.permute(*self.perm).contiguous().view(self.shape)
        self.assertTrue(torch.equal(output, expected_output))

    def test_transpose_first_false(self):
        output = npu_confusion_transpose(self.data, self.perm, self.shape, False)
        expected_output = self.data.view(self.shape).permute(*self.perm)
        self.assertTrue(torch.equal(output, expected_output))

    def test_backward_transpose_first_true(self):
        grad = torch.rand(4, 2, 3)
        result = npu_confusion_transpose_backward(grad, self.perm, self.shape, True)
        expected_result = grad.permute(*[1, 2, 0]).reshape(4, 2, 3)  # 反向计算期望的结果
        self.assertTrue(torch.equal(result, expected_result))

    def test_backward_transpose_first_false(self):
        grad = torch.rand(4, 2, 3)
        result = npu_confusion_transpose_backward(grad, self.perm, self.shape, False)
        expected_result = grad.reshape(3, 4, 2).permute(*[1, 2, 0])  # 反向计算期望的结果
        self.assertTrue(torch.equal(result, expected_result))
