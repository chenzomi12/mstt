# coding=utf-8
import unittest
from msprobe.pytorch.api_accuracy_checker.run_ut.run_overflow_check import *


class TestRunOverflowCheck(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_check_tensor_overflow_tensor_inf(self):
        x = torch.tensor(float('inf'))
        self.assertTrue(check_tensor_overflow(x))

    def test_check_tensor_overflow_tensor_nan(self):
        x = torch.tensor(float('nan'))
        self.assertTrue(check_tensor_overflow(x))

    def test_check_tensor_overflow_tensor_no_overflow(self):
        x = torch.tensor(1.0)
        self.assertFalse(check_tensor_overflow(x))

    def test_check_tensor_overflow_non_tensor_overflow(self):
        x = float('inf')
        self.assertTrue(check_tensor_overflow(x))

    def test_check_tensor_overflow_non_tensor_no_overflow(self):
        x = 1.0
        self.assertFalse(check_tensor_overflow(x))

    def test_check_tensor_overflow_bool_no_overflow(self):
        x = True
        self.assertFalse(check_tensor_overflow(x))

    def test_check_tensor_overflow_int_no_overflow(self):
        x = 42
        self.assertFalse(check_tensor_overflow(x))

    def test_check_data_overflow_list_with_overflow(self):
        tensor_overflow = torch.tensor(float('inf'))
        tensor_list = [tensor_overflow, torch.tensor(1.0)]
        self.assertTrue(check_data_overflow(tensor_list, self.device))

    def test_check_data_overflow_list_without_overflow(self):
        tensor_list = [torch.tensor(1.0), torch.tensor(2.0)]
        self.assertFalse(check_data_overflow(tensor_list, self.device))

    def test_check_data_overflow_tuple_with_overflow(self):
        tensor_overflow = torch.tensor(float('inf'))
        tensor_tuple = (tensor_overflow, torch.tensor(1.0))
        self.assertTrue(check_data_overflow(tensor_tuple, self.device))

    def test_check_data_overflow_tuple_without_overflow(self):
        tensor_tuple = (torch.tensor(1.0), torch.tensor(2.0))
        self.assertFalse(check_data_overflow(tensor_tuple, self.device))

    def test_check_data_overflow_single_tensor_overflow(self):
        tensor_overflow = torch.tensor(float('inf'))
        self.assertTrue(check_data_overflow(tensor_overflow, self.device))

    def test_check_data_overflow_single_tensor_no_overflow(self):
        tensor = torch.tensor(1.0)
        self.assertFalse(check_data_overflow(tensor, self.device))

    def test_check_data_overflowt_empty_list(self):
        empty_list = []
        self.assertFalse(check_data_overflow(empty_list, self.device))

    def test_check_data_overflow_empty_tuple(self):
        empty_tuple = ()
        self.assertFalse(check_data_overflow(empty_tuple, self.device))
