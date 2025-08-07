import unittest
from unittest.mock import Mock
from msprobe.core.data_dump.scope import ScopeFactory, ListScope, APIRangeScope, \
    ModuleRangeScope, MixRangeScope, BaseScope, RangeScope, ScopeException
from msprobe.core.common.const import Const


class TestScopeFactory(unittest.TestCase):
    def setUp(self):
        self.config = Mock()
        self.config.task = None
        self.config.level = None
        self.config.scope = None
        self.config.list = None

    def test_build_scope_none(self):
        factory = ScopeFactory(self.config)
        self.assertIsNone(factory.build_scope())

    def test_build_scope_free_benchmark(self):
        self.config.task = Const.FREE_BENCHMARK
        self.config.scope = ['scope1']
        factory = ScopeFactory(self.config)
        result = factory.build_scope()
        self.assertIsInstance(result, ListScope)

        self.config.scope = ['scope1']
        self.config.list = ['api1']
        factory = ScopeFactory(self.config)
        with self.assertRaises(ScopeException):
            factory.build_scope()


class TestBaseScope(unittest.TestCase):

    def setUp(self):
        class BaseScopeImpl(BaseScope):
            def check(self, name):
                pass
        self.base_scope = BaseScopeImpl(scope=["TestScope"], api_list=[])

    def test_rectify_args(self):
        scope = []
        api_list = "invalid_api_list"
        with self.assertRaises(ScopeException) as context:
            BaseScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.InvalidApiStr)

        api_list = [1, 2, 3]
        with self.assertRaises(ScopeException) as context:
            BaseScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.InvalidApiStr)

        scope = "module1"
        api_list = []

        expected_scope = ["module1"]
        expected_api_list = []
        result_scope, result_api_list = BaseScope.rectify_args(scope, api_list)
        self.assertEqual(result_scope, expected_scope)
        self.assertEqual(result_api_list, expected_api_list)

        scope = 123
        api_list = []
        with self.assertRaises(ScopeException) as context:
            BaseScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.InvalidScope)

        scope = ["module1", 2, "module3"]
        api_list = []
        with self.assertRaises(ScopeException) as context:
            BaseScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.InvalidScope)

    def test_check_api_list_empty(self):
        self.base_scope.api_list = []
        self.assertTrue(self.base_scope.check_api_list(""))

    def test_check_api_list_match(self):
        self.base_scope.api_list = ["api1"]
        self.assertTrue(self.base_scope.check_api_list("api1"))

    def test_check_api_list_no_match(self):
        self.base_scope.api_list = ["api1"]
        self.assertFalse(self.base_scope.check_api_list("api2"))


class TestListScope(unittest.TestCase):

    def test_rectify_args(self):
        scope = ["module1"]
        api_list = ["api1"]
        with self.assertRaises(ScopeException) as context:
            ListScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.ArgConflict)

    def test_check(self):
        list_scope = ListScope([], [])
        module_name = "module1"
        result = list_scope.check(module_name)
        self.assertTrue(result)

        list_scope = ListScope(["module1"], [])
        module_name = "module1"
        result = list_scope.check(module_name)
        self.assertTrue(result)

        list_scope = ListScope(["module1"], [])
        module_name = "module2"
        result = list_scope.check(module_name)
        self.assertFalse(result)


class MockRangeScope(RangeScope):
    def check_scope_is_valid(self):
        pass

    def check(self, name):
        pass


class TestRangeScope(unittest.TestCase):
    def test_init_valid(self):
        scope = ['Tensor.add.0.forward', 'Tensor.add.0.backward']
        rs = MockRangeScope(scope, [], Const.LEVEL_L1)
        self.assertFalse(rs.in_scope)
        self.assertFalse(rs.in_list)

    def test_rectify_args_valid(self):
        valid_scope = ['Tensor.add.0.forward', 'Tensor.add.0.backward']
        valid_api_list = ["relu"]
        rs = MockRangeScope(valid_scope, valid_api_list)
        scope, api_list = rs.rectify_args(valid_scope, valid_api_list)
        self.assertEqual(scope, valid_scope)
        self.assertEqual(api_list, valid_api_list)

    def test_rectify_args_invalid_scope_length(self):
        with self.assertRaises(ScopeException) as context:
            rs = MockRangeScope(['Tensor.add.0.forward'], [])
            rs.rectify_args(['Tensor.add.0.forward'], [])
        self.assertIn("须传入长度为2的列表", str(context.exception))

    def test_scope_length_invalid(self):
        scope = ['API.scope1.forward']
        with self.assertRaises(ScopeException):
            MockRangeScope(scope, [], Const.LEVEL_L1)

    def test_rectify_args_invalid_api_scope_format(self):
        with self.assertRaises(ScopeException) as context:
            rs = MockRangeScope(['Tensor.add.', 'API.scope2.backward'], [], Const.LEVEL_L1)
            rs.rectify_args(['Tensor.add.', 'API.scope2.backward'], [])
        self.assertIn("scope参数格式错误", str(context.exception))

    def test_rectify_args_invalid_module_scope_format(self):
        with self.assertRaises(ScopeException) as context:
            rs = MockRangeScope(['Cell.conv2d.', 'Module.scope2.backward'], [], Const.LEVEL_L0)
            rs.rectify_args(['Cell.conv2d.', 'Module.scope2.backward'], [])
        self.assertIn("scope参数格式错误", str(context.exception))


class TestAPIRangeScope(unittest.TestCase):
    
    def setUp(self):
        self.scope_valid = ['api.1', 'api.2']
        self.scope_invalid = ['Module.1', 'api.2']
        self.api_range_scope_valid = APIRangeScope(self.scope_valid, [])
        self.api_range_scope_invalid = APIRangeScope(self.scope_invalid, [])

    def test_check_scope_is_valid_valid_scope(self):
        result = self.api_range_scope_valid.check_scope_is_valid()
        self.assertTrue(result)

    def test_check_scope_is_valid_invalid_scope(self):
        result = self.api_range_scope_invalid.check_scope_is_valid()
        self.assertFalse(result)

    def test_check_api_in_scope(self):
        self.api_range_scope_valid.check_api_list = lambda x: x == 'api.1'
        self.assertTrue(self.api_range_scope_valid.check('api.1'))
        self.assertFalse(self.api_range_scope_valid.check('api.unknown'))

    def test_check_api_out_of_scope(self):
        self.api_range_scope_valid.check_api_list = lambda x: x == 'api.2'
        self.api_range_scope_valid.check('api.1')
        self.assertTrue(self.api_range_scope_valid.check('api.2'))

    def test_check_api_transition(self):
        self.api_range_scope_valid.check_api_list = lambda x: True
        self.api_range_scope_valid.check('api.1')
        self.assertTrue(self.api_range_scope_valid.in_scope)
        self.api_range_scope_valid.check('api.2')
        self.assertFalse(self.api_range_scope_valid.in_scope)


class TestModuleRangeScope(unittest.TestCase):

    def test_check_scope_is_valid(self):
        module_range_scope = ModuleRangeScope([], [])
        result = module_range_scope.check_scope_is_valid()
        self.assertTrue(result)

        module_range_scope = ModuleRangeScope(["Module.1", "Module.2"], ["Module.2"])
        self.assertTrue(module_range_scope.check_scope_is_valid())

    def test_begin_module(self):
        module_range_scope = ModuleRangeScope([], [])
        self.assertIsNone(module_range_scope.begin_module(""))

        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module1"
        module_range_scope.begin_module(module_name)
        self.assertTrue(module_range_scope.in_scope)

        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module3"
        module_range_scope.begin_module(module_name)
        self.assertFalse(module_range_scope.in_scope)

    def test_end_module(self):
        module_range_scope = ModuleRangeScope([], [])
        self.assertIsNone(module_range_scope.end_module(""))

        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module2"
        module_range_scope.in_scope = True
        module_range_scope.end_module(module_name)
        self.assertFalse(module_range_scope.in_scope)

        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module3"
        module_range_scope.in_scope = True
        module_range_scope.end_module(module_name)
        self.assertTrue(module_range_scope.in_scope)

    def test_check(self):
        module_range_scope = ModuleRangeScope([], [])
        module_name = "module1"
        result = module_range_scope.check(module_name)
        self.assertTrue(result)

        module_range_scope = ModuleRangeScope(["Module.1", "Module.2"], [])
        self.assertFalse(module_range_scope.check(""))


class TestMixRangeScope(unittest.TestCase):
    def setUp(self):
        self.scope = ['module1', 'module2']
        self.api_list = ['api1', 'api2']
        self.rs = MixRangeScope(self.scope, self.api_list)

    def test_check_scope_is_valid_with_non_empty_scope(self):
        self.assertTrue(self.rs.check_scope_is_valid())

    def test_check_scope_is_valid_with_empty_scope(self):
        rs_empty = MixRangeScope([], self.api_list)
        self.assertFalse(rs_empty.check_scope_is_valid())

    def test_begin_module_with_scope_match(self):
        self.rs.begin_module('module1')
        self.assertTrue(self.rs.in_scope)

    def test_begin_module_with_api_list_match(self):
        self.rs.begin_module('api1')
        self.assertTrue(self.rs.in_list)

    def test_end_module_with_scope_match(self):
        self.rs.end_module('module2')
        self.assertFalse(self.rs.in_scope)

    def test_end_module_with_api_list_match(self):
        self.rs.begin_module('api1')  
        self.rs.end_module('api1')
        self.assertFalse(self.rs.in_list)

    def test_check_api_list_empty(self):
        rs_empty = MixRangeScope(self.scope, [])
        self.assertTrue(rs_empty.check_api_list('any_api'))

    def test_check_api_list_match(self):
        self.assertTrue(self.rs.check_api_list('api1'))

    def test_check_api_list_no_match(self):
        self.assertFalse(self.rs.check_api_list('api3'))

    def test_check_with_scope_none_or_in_scope_true(self):
        self.rs.in_scope = True
        self.assertTrue(self.rs.check('api1'))
        self.assertFalse(self.rs.check('api3'))

    def test_check_with_scope_non_empty_and_in_scope_false(self):
        self.rs.in_scope = False
        self.assertFalse(self.rs.check('api1'))


if __name__ == '__main__':
    unittest.main()