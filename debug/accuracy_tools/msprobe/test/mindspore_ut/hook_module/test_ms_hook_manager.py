# Copyright (c) 2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

import unittest
from unittest.mock import MagicMock, patch

from msprobe.core.common.const import Const
from msprobe.core.hook_manager import HookSet, BaseHookManager
from msprobe.mindspore.dump.hook_cell.ms_hook_manager import MindsproeHookManager


class TestMindsproeHookManager(unittest.TestCase):
    def setUp(self):
        self.mock_data_collector = MagicMock()
        self.mock_config = MagicMock()
        self.mock_config.data_mode = ["all"]
        self.mock_config.task = "statistics"
        self.mock_config.level = Const.LEVEL_L1
        self.manager = MindsproeHookManager(
            self.mock_data_collector,
            self.mock_config
        )
        BaseHookManager.inner_switch[threading.get_ident()] = False

    def test_properties(self):
        self.assertIsNone(self.manager._is_recompute)

        with patch('msprobe.mindspore.dump.hook_cell.ms_hook_manager._no_grad') as mock_no_grad:
            ctx = self.manager._no_grad_context()
            mock_no_grad.assert_called_once()

    def test_add_count(self):
        with patch('msprobe.mindspore.dump.hook_cell.ms_hook_manager.HOOKCell.add_cell_count') as mock_add:
            self.manager._add_count("test_module")
            mock_add.assert_called_once_with("test_module")

    def test_process_kwargs_and_output(self):
        mock_module = MagicMock()
        mock_module.msprobe_input_kwargs = {"kw1": "v1"}

        kwargs, output = self.manager._process_kwargs_and_output(
            mock_module, Const.API, "output_value", "ignored"
        )
        self.assertEqual(kwargs, {"kw1": "v1"})
        self.assertEqual(output, "output_value")

        with patch('msprobe.mindspore.dump.hook_cell.ms_hook_manager.has_kwargs_in_forward_hook', return_value=True):
            kwargs, output = self.manager._process_kwargs_and_output(
                mock_module, Const.MODULE, "kwargs_value", "output_value"
            )
            self.assertEqual(kwargs, "kwargs_value")
            self.assertEqual(output, "output_value")

    def test_build_hook(self):
        hookset = self.manager.build_hook(Const.API, "test_api")
        self.assertIsInstance(hookset, HookSet)
        self.assertEqual(hookset.forward_hook.__name__, "forward_hook")
        self.assertEqual(hookset.forward_pre_hook.__name__, "forward_pre_hook")
        self.assertEqual(hookset.backward_hook.__name__, "backward_hook")
        self.assertEqual(hookset.backward_pre_hook.__name__, "backward_pre_hook")
        hookset = self.manager.build_hook(Const.MODULE, "test_module")
        self.assertEqual(hookset.forward_pre_hook.__name__, "forward_pre_hook")

    def test_need_exchange(self):
        mock_module = MagicMock()
        del mock_module.has_pre_hook_called
        self.assertFalse(self.manager._need_exchange(mock_module))

        mock_module.has_pre_hook_called = False
        self.assertFalse(self.manager._need_exchange(mock_module))

        mock_module.has_pre_hook_called = True
        self.assertTrue(self.manager._need_exchange(mock_module))

    def test_get_params_dict(self):
        mock_module = MagicMock()

        self.mock_config.task = Const.STRUCTURE
        params_dict = self.manager._get_params_dict(mock_module)
        self.assertEqual(params_dict, {})

        self.mock_config.task = "statistics"
        mock_params = {
            "test_module.weight": "w1",
            "test_module.bias": "b1"
        }
        mock_module.parameters_dict.return_value = mock_params
        params_dict = self.manager._get_params_dict(mock_module)
        mock_module.parameters_dict.assert_called_once_with(recurse=False)
        self.assertEqual(params_dict, {"weight": "w1", "bias": "b1"})

    def test_build_backward_pre_hook(self):
        hook_fn = self.manager._build_backward_pre_hook(Const.MODULE, "test_module_backward")

        mock_module = MagicMock()
        mock_grad_input = ("grad1", "grad2")

        with patch.object(self.manager, '_should_execute_hook', return_value=False):
            hook_fn(mock_module, mock_grad_input)
            self.mock_data_collector.backward_input_data_collect.assert_not_called()

        self.mock_config.level = Const.LEVEL_L2
        with patch.object(self.manager, '_should_execute_hook', return_value=True):
            hook_fn(mock_module, mock_grad_input)

            self.mock_data_collector.update_api_or_module_name.assert_called_with("test_module_backward")
            self.mock_data_collector.backward_input_data_collect.assert_called_once()

            call_args = self.mock_data_collector.backward_input_data_collect.call_args[0]
            module_input = call_args[3]
            self.assertEqual(module_input.grad_input, mock_grad_input)

            self.assertFalse(BaseHookManager.inner_switch[threading.get_ident()])

        self.mock_config.level = Const.LEVEL_L1
        with patch.object(self.manager, '_should_execute_hook', return_value=True):
            hook_fn(mock_module, mock_grad_input)
            self.mock_data_collector.backward_input_data_collect.assert_called_once()
