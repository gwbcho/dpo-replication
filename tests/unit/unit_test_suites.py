import os
import unittest

import tests.unit.policy.gac_networks as test_gac_networks


def create_and_run_test_suite(test_module):
    print('Running tests from: ' + str(test_module.__file__))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromModule(test_module))

    unittest.TextTestRunner(verbosity=1).run(suite)
    print('')


def main():
    create_and_run_test_suite(test_gac_networks)


if __name__ == '__main__':
    main()
