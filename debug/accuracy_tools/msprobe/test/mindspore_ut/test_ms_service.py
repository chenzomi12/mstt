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

from collections import defaultdict
import unittest
from unittest.mock import MagicMock, patch
from msprobe.mindspore.dump.jit_dump import JitDump
from msprobe.mindspore.mindspore_service import MindsporeService
from msprobe.core.common.utils import Const
from mindspore import ops

try:
    from mindspore.common._pijit_context import PIJitCaptureContext
except ImportError:
    pijit_label = False
else:
    pijit_label = True


class TestMindsporeService(unittest.TestCase):
    def setUp(self):

        self.config = MagicMock()
        self.config.step = []
        self.config.rank = []
        self.config.level_ori = Const.LEVEL_MIX
        self.config.task = Const.STATISTICS
        
        with patch('msprobe.core.service.build_data_collector'):
            self.service = MindsporeService(self.config)
        
        self.service.logger = MagicMock()
        self.service.data_collector = MagicMock()
        self.service.primitive_hook_service = MagicMock()
        self.service.cell_processor = MagicMock()
        self.service.api_register = MagicMock()
    
    @patch('msprobe.mindspore.mindspore_service.is_mindtorch')
    def test_framework_type(self, mock_is_mindtorch):
        mock_is_mindtorch.return_value = True
        self.assertEqual(self.service._get_framework_type, Const.MT_FRAMEWORK)
        mock_is_mindtorch.return_value = False
        self.assertEqual(self.service._get_framework_type, Const.MS_FRAMEWORK)
    
    @patch('msprobe.mindspore.mindspore_service.get_rank_if_initialized')
    def test_get_current_rank(self, mock_get_rank):
        mock_get_rank.return_value = 3
        self.assertEqual(MindsporeService._get_current_rank(), 3)
    
    def test_init_specific_components(self):
        with patch('msprobe.core.service.build_data_collector'):
            service = MindsporeService(self.config)
        
        self.assertIsNotNone(service.logger)
        self.assertIsNotNone(service.api_register)
        self.assertIsNotNone(service.primitive_hook_service)
        self.assertIsNotNone(service.cell_processor)
        self.assertIsNotNone(service.hook_manager)
    
    @patch.object(JitDump, "set_data_collector")
    @patch.object(JitDump, "set_config")
    @patch('msprobe.mindspore.mindspore_service.ms.common.api')
    def test_setup_jit_context_with_pijit(self, mock_ms_api, mock_jit_set_config, mock_set_data_collector):
        mock_ms_api.__dict__['_MindsporeFunctionExecutor'] = MagicMock()
        self.service._setup_jit_context()

        mock_jit_set_config.assert_called_once_with(self.config)
        mock_set_data_collector.assert_called_once_with(self.service.data_collector)
        self.assertEqual(mock_ms_api._MindsporeFunctionExecutor, JitDump)
        self.assertEqual(mock_ms_api._PyNativeExecutor.grad, JitDump.grad)
        if pijit_label:
            self.assertEqual(PIJitCaptureContext.__enter__, self.service.empty)
            self.assertEqual(PIJitCaptureContext.__exit__, self.service.empty)
    
    @patch('msprobe.mindspore.mindspore_service.JitDump')
    def test_change_jit_switch(self, mock_jit_dump):
        self.service._change_jit_switch(True)
        self.assertTrue(mock_jit_dump.jit_dump_switch)
        
        self.service._change_jit_switch(False)
        self.assertFalse(mock_jit_dump.jit_dump_switch)
    
    def test_register_module_hook(self):
        model_mock = MagicMock()
        self.service.model = model_mock
        self.service._register_module_hook()
        
        self.service.cell_processor.register_cell_hook.assert_called_once_with(
            model_mock, self.service.build_hook, self.config
        )
    
    def test_register_primitive_hook(self):
        self.service.config.level = Const.LEVEL_MIX
        primitive_attr = ops.Add()
        primitive_name = "primitive_api"
        mock_model = MagicMock()
        cell_mock = MagicMock()
        cell_mock.primitive_api = primitive_attr
        primitive_combined_name = primitive_name + Const.SEP + primitive_attr.__class__.__name__
        self.service.model = mock_model
        with patch('msprobe.mindspore.mindspore_service.get_cells_and_names_with_index') as mock_get_cells_and_names:
            mock_get_cells_and_names.return_value = ({'-1': [("cell_name", cell_mock)]}, {})
            self.service._register_primitive_hook()
        self.assertTrue(hasattr(primitive_attr.__class__, '__call__'))
        self.assertEqual(self.service.primitive_hook_service.wrap_primitive.call_args[0][1],
                         primitive_combined_name)
    
    def test_reset_status(self):
        self.service.primitive_hook_service.primitive_counters = defaultdict(int)
        self.service.primitive_hook_service.primitive_counters['test_prim'] = 5
        self.service._reset_status()
        self.assertEqual(self.service.primitive_hook_service.primitive_counters, {})
        with patch('msprobe.mindspore.mindspore_service.JitDump') as mock_jit_dump:
            mock_jit_dump.jit_count = defaultdict(int)
            mock_jit_dump.jit_count['test_jit'] = 3
            self.service._reset_status()
            self.assertEqual(mock_jit_dump.jit_count, {})
    
    @patch('msprobe.mindspore.mindspore_service.JitDump')
    def test_start_jit_enabled(self, mock_jit_dump):
        self.service.data_collector.data_processor.is_terminated = False
        model_mock = MagicMock()
        self.service.start(model=model_mock)
        self.assertTrue(mock_jit_dump.jit_dump_switch)
    
    @patch('msprobe.mindspore.mindspore_service.JitDump')
    def test_stop_jit_disabled(self, mock_jit_dump):
        self.config.level = Const.LEVEL_MIX
        self.service.current_iter = 1
        self.service.current_rank = 0
        
        self.service.stop()
        
        self.assertFalse(mock_jit_dump.jit_dump_switch)
    
    @patch('msprobe.mindspore.mindspore_service.JitDump')
    @patch('msprobe.mindspore.mindspore_service.ms.common.api')
    def test_setup_jit_context_level_not_supported(self, mock_ms_api, mock_jit_dump):
        self.service.config.level = Const.LEVEL_DEBUG
        
        self.service._setup_jit_context()
        
        mock_jit_dump.set_config.assert_not_called()
        mock_jit_dump.set_data_collector.assert_not_called()
