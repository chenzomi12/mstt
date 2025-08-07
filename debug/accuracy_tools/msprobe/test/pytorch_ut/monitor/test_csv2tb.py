# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import os
import shutil
import random
import unittest
import torch
import numpy as np
import torch.nn as nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from msprobe.pytorch import TrainerMon
from msprobe.core.common.const import MonitorConst
from msprobe.pytorch.monitor.csv2tb import parse_step_fn, csv2tensorboard_by_step
from msprobe.pytorch.hook_module.api_register import get_api_register

get_api_register().restore_all_api()

base_dir = os.path.dirname(os.path.realpath(__file__))
config_json_path = os.path.join(base_dir, "config", "all_config.json")
monitor_output = os.path.join(base_dir, "./monitor_output_csv2tb")


def seed_all(seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)


seed_all()

inputs = [torch.rand(10, 10) for _ in range(10)]
labels = [torch.randint(0, 5, (10,)) for _ in range(10)]


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.relu(x1)
        return x2


def extract_scalars_from_tensorboard(log_dir):
    # 初始化 EventAccumulator
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()  # 加载事件数据

    # 获取所有 scalar 标签
    scalar_tags = event_acc.Tags()['scalars']

    # 构建字典，键为标签，值为对应的 (step, value) 列表
    scalars_dict = {}
    for tag in scalar_tags:
        scalar_events = event_acc.Scalars(tag)
        scalars_dict[tag] = [(event.step, event.value) for event in scalar_events]

    return scalars_dict


def dict_equal(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b

    if set(a.keys()) != set(b.keys()):
        return False

    for key in a:
        if not dict_equal(a[key], b[key]):
            return False

    return True


def compare_scalar_dicts(dict1, dict2):
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    for key in dict1:
        list1 = dict1[key]
        list2 = dict2[key]

        if len(list1) != len(list2):
            return False

        # 对比每对 (step, value)
        for (step1, value1), (step2, value2) in zip(list1, list2):
            if step1 != step2:
                return False

            if not (value1 == value2 or (np.isnan(value1) and np.isnan(value2))):
                return False
    return True


class TestGradMonitor(unittest.TestCase):
    timestamp_dirpath = None
    csv2tb_dirpath = None

    @classmethod
    def setUpClass(cls):

        os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = monitor_output
        if os.path.exists(monitor_output):
            shutil.rmtree(monitor_output)

        loss_fun = nn.CrossEntropyLoss()
        test_module = MockModule()
        nn.init.constant_(test_module.linear.weight, 1.0)
        nn.init.constant_(test_module.linear.bias, 1.0)
        optimizer = torch.optim.Adam(test_module.parameters())

        monitor = TrainerMon(config_json_path, params_have_main_grad=False)
        monitor.set_monitor(test_module, grad_acc_steps=1, optimizer=optimizer)

        for input_data, label in zip(inputs, labels):
            output = test_module(input_data)
            loss = loss_fun(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cls.timestamp_dirpath = os.path.join(monitor_output, os.listdir(monitor_output)[0])
        csv2tensorboard_by_step(monitor_output)
        for dirname in os.listdir(monitor_output):
            if "csv2tensorboard" in dirname:
                cls.csv2tb_dirpath = os.path.join(monitor_output, dirname, "rank0")
        os.environ.pop(MonitorConst.MONITOR_OUTPUT_DIR)

    def setUp(self):
        self.maxDiff = None

    def test_actv(self):
        data = parse_step_fn(os.path.join(self.timestamp_dirpath, "actv_0-2.csv"))
        result = {
            'vp0:.input:micro0': {
                0: {'nans': 0.0, 'norm': 5.550016},
                1: {'nans': 0.0, 'norm': 5.975112},
                2: {'nans': 0.0, 'norm': 5.789881}
            },
            'vp0:.output:micro0': {
                0: {'nans': 0.0, 'norm': 41.842655},
                1: {'nans': 0.0, 'norm': 44.40981},
                2: {'nans': 0.0, 'norm': 43.578354}
            },
            'vp0:linear.input:micro0': {
                0: {'nans': 0.0, 'norm': 5.550016},
                1: {'nans': 0.0, 'norm': 5.975112},
                2: {'nans': 0.0, 'norm': 5.789881}
            },
            'vp0:linear.output:micro0': {
                0: {'nans': 0.0, 'norm': 41.842655},
                1: {'nans': 0.0, 'norm': 44.40981},
                2: {'nans': 0.0, 'norm': 43.578354}
            },
            'vp0:relu.input:micro0': {
                0: {'nans': 0.0, 'norm': 41.842655},
                1: {'nans': 0.0, 'norm': 44.40981},
                2: {'nans': 0.0, 'norm': 43.578354}
            },
            'vp0:relu.output:micro0': {
                0: {'nans': 0.0, 'norm': 41.842655},
                1: {'nans': 0.0, 'norm': 44.40981},
                2: {'nans': 0.0, 'norm': 43.578354}
            }
        }
        self.assertDictEqual(data, result)
        tb_data = extract_scalars_from_tensorboard(os.path.join(self.csv2tb_dirpath, "actv"))
        print(tb_data)
        tb_result = {
            'vp0:.input:micro0/nans': [(0, 0.0),
                                       (1, 0.0),
                                       (2, 0.0),
                                       (3, 0.0),
                                       (4, 0.0),
                                       (5, 0.0),
                                       (6, 0.0),
                                       (7, 0.0),
                                       (8, 0.0),
                                       (9, 0.0)],
            'vp0:.input:micro0/norm': [(0, 5.550015926361084),
                                       (1, 5.975111961364746),
                                       (2, 5.789881229400635),
                                       (3, 6.052319049835205),
                                       (4, 5.573315143585205),
                                       (5, 5.864360809326172),
                                       (6, 5.292460918426514),
                                       (7, 5.477899074554443),
                                       (8, 5.884613990783691),
                                       (9, 5.456457138061523)],
            'vp0:.output:micro0/nans': [(0, 0.0),
                                        (1, 0.0),
                                        (2, 0.0),
                                        (3, 0.0),
                                        (4, 0.0),
                                        (5, 0.0),
                                        (6, 0.0),
                                        (7, 0.0),
                                        (8, 0.0),
                                        (9, 0.0)],
            'vp0:.output:micro0/norm': [(0, 41.842655181884766),
                                        (1, 44.40980911254883),
                                        (2, 43.57835388183594),
                                        (3, 45.83631134033203),
                                        (4, 42.0673828125),
                                        (5, 43.46839141845703),
                                        (6, 39.77947235107422),
                                        (7, 40.200843811035156),
                                        (8, 44.453147888183594),
                                        (9, 40.841522216796875)],
            'vp0:linear.input:micro0/nans': [(0, 0.0),
                                             (1, 0.0),
                                             (2, 0.0),
                                             (3, 0.0),
                                             (4, 0.0),
                                             (5, 0.0),
                                             (6, 0.0),
                                             (7, 0.0),
                                             (8, 0.0),
                                             (9, 0.0)],
            'vp0:linear.input:micro0/norm': [(0, 5.550015926361084),
                                             (1, 5.975111961364746),
                                             (2, 5.789881229400635),
                                             (3, 6.052319049835205),
                                             (4, 5.573315143585205),
                                             (5, 5.864360809326172),
                                             (6, 5.292460918426514),
                                             (7, 5.477899074554443),
                                             (8, 5.884613990783691),
                                             (9, 5.456457138061523)],
            'vp0:linear.output:micro0/nans': [(0, 0.0),
                                              (1, 0.0),
                                              (2, 0.0),
                                              (3, 0.0),
                                              (4, 0.0),
                                              (5, 0.0),
                                              (6, 0.0),
                                              (7, 0.0),
                                              (8, 0.0),
                                              (9, 0.0)],
            'vp0:linear.output:micro0/norm': [(0, 41.842655181884766),
                                              (1, 44.40980911254883),
                                              (2, 43.57835388183594),
                                              (3, 45.83631134033203),
                                              (4, 42.0673828125),
                                              (5, 43.46839141845703),
                                              (6, 39.77947235107422),
                                              (7, 40.200843811035156),
                                              (8, 44.453147888183594),
                                              (9, 40.841522216796875)],
            'vp0:relu.input:micro0/nans': [(0, 0.0),
                                           (1, 0.0),
                                           (2, 0.0),
                                           (3, 0.0),
                                           (4, 0.0),
                                           (5, 0.0),
                                           (6, 0.0),
                                           (7, 0.0),
                                           (8, 0.0),
                                           (9, 0.0)],
            'vp0:relu.input:micro0/norm': [(0, 41.842655181884766),
                                           (1, 44.40980911254883),
                                           (2, 43.57835388183594),
                                           (3, 45.83631134033203),
                                           (4, 42.0673828125),
                                           (5, 43.46839141845703),
                                           (6, 39.77947235107422),
                                           (7, 40.200843811035156),
                                           (8, 44.453147888183594),
                                           (9, 40.841522216796875)],
            'vp0:relu.output:micro0/nans': [(0, 0.0),
                                            (1, 0.0),
                                            (2, 0.0),
                                            (3, 0.0),
                                            (4, 0.0),
                                            (5, 0.0),
                                            (6, 0.0),
                                            (7, 0.0),
                                            (8, 0.0),
                                            (9, 0.0)],
            'vp0:relu.output:micro0/norm': [(0, 41.842655181884766),
                                            (1, 44.40980911254883),
                                            (2, 43.57835388183594),
                                            (3, 45.83631134033203),
                                            (4, 42.0673828125),
                                            (5, 43.46839141845703),
                                            (6, 39.77947235107422),
                                            (7, 40.200843811035156),
                                            (8, 44.453147888183594),
                                            (9, 40.841522216796875)]}
        self.assertDictEqual(tb_data, tb_result)

    def test_actv_grad(self):
        data = parse_step_fn(os.path.join(self.timestamp_dirpath, "actv_grad_0-2.csv"))
        nan = np.nan
        result = {
            'vp0:.input:micro0': {
                0: {'norm': nan, 'nans': nan},
                1: {'norm': nan, 'nans': nan},
                2: {'norm': nan, 'nans': nan}
            },
            'vp0:.output:micro0': {
                0: {'norm': 0.282843, 'nans': 0.0},
                1: {'norm': 0.282617, 'nans': 0.0},
                2: {'norm': 0.282655, 'nans': 0.0}
            },
            'vp0:relu.input:micro0': {
                0: {'norm': 0.282843, 'nans': 0.0},
                1: {'norm': 0.282617, 'nans': 0.0},
                2: {'norm': 0.282655, 'nans': 0.0}
            },
            'vp0:relu.output:micro0': {
                0: {'norm': 0.282843, 'nans': 0.0},
                1: {'norm': 0.282617, 'nans': 0.0},
                2: {'norm': 0.282655, 'nans': 0.0}
            },
            'vp0:linear.input:micro0': {
                0: {'norm': nan, 'nans': nan},
                1: {'norm': nan, 'nans': nan},
                2: {'norm': nan, 'nans': nan}
            },
            'vp0:linear.output:micro0': {
                0: {'norm': 0.282843, 'nans': 0.0},
                1: {'norm': 0.282617, 'nans': 0.0},
                2: {'norm': 0.282655, 'nans': 0.0}
            }
        }
        print(data)

        tb_data = extract_scalars_from_tensorboard(os.path.join(self.csv2tb_dirpath, "actv_grad"))
        tb_result = {
            'vp0:.input:micro0/nans': [(0, nan),
                                       (1, nan),
                                       (2, nan),
                                       (3, nan),
                                       (4, nan),
                                       (5, nan),
                                       (6, nan),
                                       (7, nan),
                                       (8, nan),
                                       (9, nan)],
            'vp0:.input:micro0/norm': [(0, nan),
                                       (1, nan),
                                       (2, nan),
                                       (3, nan),
                                       (4, nan),
                                       (5, nan),
                                       (6, nan),
                                       (7, nan),
                                       (8, nan),
                                       (9, nan)],
            'vp0:.output:micro0/nans': [(0, 0.0),
                                        (1, 0.0),
                                        (2, 0.0),
                                        (3, 0.0),
                                        (4, 0.0),
                                        (5, 0.0),
                                        (6, 0.0),
                                        (7, 0.0),
                                        (8, 0.0),
                                        (9, 0.0)],
            'vp0:.output:micro0/norm': [(0, 0.2828429937362671),
                                        (1, 0.2826170027256012),
                                        (2, 0.2826550006866455),
                                        (3, 0.2828519940376282),
                                        (4, 0.2822929918766022),
                                        (5, 0.2826640009880066),
                                        (6, 0.28316599130630493),
                                        (7, 0.28274500370025635),
                                        (8, 0.2833530008792877),
                                        (9, 0.2825529873371124)],
            'vp0:linear.input:micro0/nans': [(0, nan),
                                             (1, nan),
                                             (2, nan),
                                             (3, nan),
                                             (4, nan),
                                             (5, nan),
                                             (6, nan),
                                             (7, nan),
                                             (8, nan),
                                             (9, nan)],
            'vp0:linear.input:micro0/norm': [(0, nan),
                                             (1, nan),
                                             (2, nan),
                                             (3, nan),
                                             (4, nan),
                                             (5, nan),
                                             (6, nan),
                                             (7, nan),
                                             (8, nan),
                                             (9, nan)],
            'vp0:linear.output:micro0/nans': [(0, 0.0),
                                              (1, 0.0),
                                              (2, 0.0),
                                              (3, 0.0),
                                              (4, 0.0),
                                              (5, 0.0),
                                              (6, 0.0),
                                              (7, 0.0),
                                              (8, 0.0),
                                              (9, 0.0)],
            'vp0:linear.output:micro0/norm': [(0, 0.2828429937362671),
                                              (1, 0.2826170027256012),
                                              (2, 0.2826550006866455),
                                              (3, 0.2828519940376282),
                                              (4, 0.2822929918766022),
                                              (5, 0.2826640009880066),
                                              (6, 0.28316599130630493),
                                              (7, 0.28274500370025635),
                                              (8, 0.2833530008792877),
                                              (9, 0.2825529873371124)],
            'vp0:relu.input:micro0/nans': [(0, 0.0),
                                           (1, 0.0),
                                           (2, 0.0),
                                           (3, 0.0),
                                           (4, 0.0),
                                           (5, 0.0),
                                           (6, 0.0),
                                           (7, 0.0),
                                           (8, 0.0),
                                           (9, 0.0)],
            'vp0:relu.input:micro0/norm': [(0, 0.2828429937362671),
                                           (1, 0.2826170027256012),
                                           (2, 0.2826550006866455),
                                           (3, 0.2828519940376282),
                                           (4, 0.2822929918766022),
                                           (5, 0.2826640009880066),
                                           (6, 0.28316599130630493),
                                           (7, 0.28274500370025635),
                                           (8, 0.2833530008792877),
                                           (9, 0.2825529873371124)],
            'vp0:relu.output:micro0/nans': [(0, 0.0),
                                            (1, 0.0),
                                            (2, 0.0),
                                            (3, 0.0),
                                            (4, 0.0),
                                            (5, 0.0),
                                            (6, 0.0),
                                            (7, 0.0),
                                            (8, 0.0),
                                            (9, 0.0)],
            'vp0:relu.output:micro0/norm': [(0, 0.2828429937362671),
                                            (1, 0.2826170027256012),
                                            (2, 0.2826550006866455),
                                            (3, 0.2828519940376282),
                                            (4, 0.2822929918766022),
                                            (5, 0.2826640009880066),
                                            (6, 0.28316599130630493),
                                            (7, 0.28274500370025635),
                                            (8, 0.2833530008792877),
                                            (9, 0.2825529873371124)]
        }
        print(tb_data)

    def test_param(self):
        data = parse_step_fn(os.path.join(self.timestamp_dirpath, "param_origin_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                0: {'nans': 0.0, 'norm': 2.236068},
                1: {'nans': 0.0, 'norm': 2.236198},
                2: {'nans': 0.0, 'norm': 2.235769}
            },
            'vp0:linear.weight': {
                0: {'nans': 0.0, 'norm': 7.071068},
                1: {'nans': 0.0, 'norm': 7.068808},
                2: {'nans': 0.0, 'norm': 7.06771}
            }
        }
        self.assertDictEqual(data, result)
        tb_data = extract_scalars_from_tensorboard(os.path.join(self.csv2tb_dirpath, "param_origin"))
        tb_result = {
            'vp0:linear.weight/norm': [
                (0, 7.071067810058594),
                (1, 7.068808078765869),
                (2, 7.067709922790527),
                (3, 7.0673418045043945),
                (4, 7.066926956176758),
                (5, 7.066311836242676),
                (6, 7.065629959106445),
                (7, 7.065262794494629),
                (8, 7.065001964569092),
                (9, 7.064840793609619)],
            'vp0:linear.weight/nans': [
                (0, 0.0),
                (1, 0.0),
                (2, 0.0),
                (3, 0.0),
                (4, 0.0),
                (5, 0.0),
                (6, 0.0),
                (7, 0.0),
                (8, 0.0),
                (9, 0.0)],
            'vp0:linear.bias/norm': [
                (0, 2.2360680103302),
                (1, 2.2361979484558105),
                (2, 2.235769033432007),
                (3, 2.235903024673462),
                (4, 2.2360129356384277),
                (5, 2.2359039783477783),
                (6, 2.2357990741729736),
                (7, 2.2357349395751953),
                (8, 2.2356700897216797),
                (9, 2.235619068145752)
            ],
            'vp0:linear.bias/nans': [
                (0, 0.0),
                (1, 0.0),
                (2, 0.0),
                (3, 0.0),
                (4, 0.0),
                (5, 0.0),
                (6, 0.0),
                (7, 0.0),
                (8, 0.0),
                (9, 0.0)
            ]
        }
        self.assertDictEqual(tb_data, tb_result)

    def test_exp_avg(self):
        data = parse_step_fn(os.path.join(self.timestamp_dirpath, "exp_avg_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                1: {'nans': 0.0, 'norm': 0.024495},
                2: {'nans': 0.0, 'norm': 0.052203}
            },
            'vp0:linear.weight': {
                1: {'nans': 0.0, 'norm': 0.052394},
                2: {'nans': 0.0, 'norm': 0.099221}
            }
        }
        self.assertDictEqual(data, result)
        tb_data = extract_scalars_from_tensorboard(os.path.join(self.csv2tb_dirpath, "exp_avg"))
        tb_result = {
            'vp0:linear.bias/nans': [(1, 0.0),
                                     (2, 0.0),
                                     (3, 0.0),
                                     (4, 0.0),
                                     (5, 0.0),
                                     (6, 0.0),
                                     (7, 0.0),
                                     (8, 0.0),
                                     (9, 0.0)],
            'vp0:linear.bias/norm': [(1, 0.024495000019669533),
                                     (2, 0.05220299959182739),
                                     (3, 0.06452500075101852),
                                     (4, 0.05751600116491318),
                                     (5, 0.07189200073480606),
                                     (6, 0.07151799649000168),
                                     (7, 0.053112998604774475),
                                     (8, 0.06187799945473671),
                                     (9, 0.04195199906826019)],
            'vp0:linear.weight/nans': [(1, 0.0),
                                       (2, 0.0),
                                       (3, 0.0),
                                       (4, 0.0),
                                       (5, 0.0),
                                       (6, 0.0),
                                       (7, 0.0),
                                       (8, 0.0),
                                       (9, 0.0)],
            'vp0:linear.weight/norm': [(1, 0.05239399895071983),
                                       (2, 0.09922099858522415),
                                       (3, 0.12258800119161606),
                                       (4, 0.11325100064277649),
                                       (5, 0.14186500012874603),
                                       (6, 0.14408400654792786),
                                       (7, 0.11372199654579163),
                                       (8, 0.12264800071716309),
                                       (9, 0.09017200022935867)]}
        self.assertDictEqual(tb_data, tb_result)

    def test_exp_avg_sq(self):
        data = parse_step_fn(os.path.join(self.timestamp_dirpath, "exp_avg_sq_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                1: {'nans': 0.0, 'norm': 4.2e-05},
                2: {'nans': 0.0, 'norm': 9.6e-05}
            },
            'vp0:linear.weight': {
                1: {'nans': 0.0, 'norm': 6.7e-05},
                2: {'nans': 0.0, 'norm': 0.000126}
            }
        }
        self.assertDictEqual(data, result)
        tb_data = extract_scalars_from_tensorboard(os.path.join(self.csv2tb_dirpath, "exp_avg_sq"))
        tb_result = {
            'vp0:linear.bias/nans': [(1, 0.0),
                                     (2, 0.0),
                                     (3, 0.0),
                                     (4, 0.0),
                                     (5, 0.0),
                                     (6, 0.0),
                                     (7, 0.0),
                                     (8, 0.0),
                                     (9, 0.0)],
            'vp0:linear.bias/norm': [(1, 4.199999966658652e-05),
                                     (2, 9.600000339560211e-05),
                                     (3, 0.00013099999341648072),
                                     (4, 0.00013099999341648072),
                                     (5, 0.00016500000492669642),
                                     (6, 0.0001900000061141327),
                                     (7, 0.00020199999562464654),
                                     (8, 0.00022899999748915434),
                                     (9, 0.00024300000222865492)],
            'vp0:linear.weight/nans': [(1, 0.0),
                                       (2, 0.0),
                                       (3, 0.0),
                                       (4, 0.0),
                                       (5, 0.0),
                                       (6, 0.0),
                                       (7, 0.0),
                                       (8, 0.0),
                                       (9, 0.0)],
            'vp0:linear.weight/norm': [(1, 6.70000008540228e-05),
                                       (2, 0.00012599999899975955),
                                       (3, 0.00015799999528098851),
                                       (4, 0.00016599999798927456),
                                       (5, 0.00021399999968707561),
                                       (6, 0.00024199999461416155),
                                       (7, 0.00026000000070780516),
                                       (8, 0.00028700000257231295),
                                       (9, 0.0003060000017285347)]}
        self.assertDictEqual(tb_data, tb_result)

    def test_grad_reduced(self):
        data = parse_step_fn(os.path.join(self.timestamp_dirpath, "grad_reduced_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                0: {'nans': 0.0, 'norm': 0.244949},
                1: {'nans': 0.0, 'norm': 0.314345},
                2: {'nans': 0.0, 'norm': 0.281475}
            },
            'vp0:linear.weight': {
                0: {'nans': 0.0, 'norm': 0.523935},
                1: {'nans': 0.0, 'norm': 0.595672},
                2: {'nans': 0.0, 'norm': 0.497603}
            }
        }
        self.assertDictEqual(data, result)
        tb_data = extract_scalars_from_tensorboard(os.path.join(self.csv2tb_dirpath, "grad_reduced"))
        tb_result = {
            'vp0:linear.bias/nans': [(0, 0.0),
                                     (1, 0.0),
                                     (2, 0.0),
                                     (3, 0.0),
                                     (4, 0.0),
                                     (5, 0.0),
                                     (6, 0.0),
                                     (7, 0.0),
                                     (8, 0.0),
                                     (9, 0.0)],
            'vp0:linear.bias/norm': [(0, 0.24494899809360504),
                                     (1, 0.31434500217437744),
                                     (2, 0.2814750075340271),
                                     (3, 0.006068999879062176),
                                     (4, 0.2398650050163269),
                                     (5, 0.2817699909210205),
                                     (6, 0.1456969976425171),
                                     (7, 0.2817710041999817),
                                     (8, 0.15226399898529053),
                                     (9, 0.1355219930410385)],
            'vp0:linear.weight/nans': [(0, 0.0),
                                       (1, 0.0),
                                       (2, 0.0),
                                       (3, 0.0),
                                       (4, 0.0),
                                       (5, 0.0),
                                       (6, 0.0),
                                       (7, 0.0),
                                       (8, 0.0),
                                       (9, 0.0)],
            'vp0:linear.weight/norm': [(0, 0.5239350199699402),
                                       (1, 0.5956720113754272),
                                       (2, 0.49760299921035767),
                                       (3, 0.23948900401592255),
                                       (4, 0.5050320029258728),
                                       (5, 0.5136330127716064),
                                       (6, 0.3642309904098511),
                                       (7, 0.4831080138683319),
                                       (8, 0.3234719932079315),
                                       (9, 0.32385098934173584)]}
        self.assertDictEqual(tb_data, tb_result)

    def test_grad_unreduced(self):
        data = parse_step_fn(os.path.join(self.timestamp_dirpath, "grad_unreduced_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                0: {'nans': 0.0, 'norm': 0.244949},
                1: {'nans': 0.0, 'norm': 0.314345},
                2: {'nans': 0.0, 'norm': 0.281475}
            },
            'vp0:linear.weight': {
                0: {'nans': 0.0, 'norm': 0.523935},
                1: {'nans': 0.0, 'norm': 0.595672},
                2: {'nans': 0.0, 'norm': 0.497603}
            }
        }
        self.assertDictEqual(data, result)

        tb_data = extract_scalars_from_tensorboard(os.path.join(self.csv2tb_dirpath, "grad_unreduced"))
        tb_result = {
            'vp0:linear.bias/nans': [(0, 0.0),
                                     (1, 0.0),
                                     (2, 0.0),
                                     (3, 0.0),
                                     (4, 0.0),
                                     (5, 0.0),
                                     (6, 0.0),
                                     (7, 0.0),
                                     (8, 0.0),
                                     (9, 0.0)],
            'vp0:linear.bias/norm': [(0, 0.24494899809360504),
                                     (1, 0.31434500217437744),
                                     (2, 0.2814750075340271),
                                     (3, 0.006068999879062176),
                                     (4, 0.2398650050163269),
                                     (5, 0.2817699909210205),
                                     (6, 0.1456969976425171),
                                     (7, 0.2817710041999817),
                                     (8, 0.15226399898529053),
                                     (9, 0.1355219930410385)],
            'vp0:linear.weight/nans': [(0, 0.0),
                                       (1, 0.0),
                                       (2, 0.0),
                                       (3, 0.0),
                                       (4, 0.0),
                                       (5, 0.0),
                                       (6, 0.0),
                                       (7, 0.0),
                                       (8, 0.0),
                                       (9, 0.0)],
            'vp0:linear.weight/norm': [(0, 0.5239350199699402),
                                       (1, 0.5956720113754272),
                                       (2, 0.49760299921035767),
                                       (3, 0.23948900401592255),
                                       (4, 0.5050320029258728),
                                       (5, 0.5136330127716064),
                                       (6, 0.3642309904098511),
                                       (7, 0.4831080138683319),
                                       (8, 0.3234719932079315),
                                       (9, 0.32385098934173584)]}
        self.assertDictEqual(tb_data, tb_result)


if __name__ == '__main__':
    unittest.main()
