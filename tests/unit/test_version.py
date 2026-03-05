# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import fvdb


class TestVersion(unittest.TestCase):
    def test_dunder_version_format(self):
        self.assertIsInstance(fvdb.__version__, str)
        self.assertRegex(fvdb.__version__, r"^\d+\.\d+\.\d+$")

    def test_version_info_tuple(self):
        self.assertIsInstance(fvdb.__version_info__, tuple)
        self.assertEqual(len(fvdb.__version_info__), 3)
        for part in fvdb.__version_info__:
            self.assertIsInstance(part, int)

    def test_version_fvdb_matches_dunder(self):
        self.assertEqual(fvdb.version.fvdb, fvdb.__version__)

    def test_version_nanovdb_format(self):
        nv = fvdb.version.nanovdb
        self.assertIsInstance(nv, str)
        self.assertRegex(nv, r"^\d+\.\d+\.\d+$", f"nanovdb version {nv!r} does not match X.Y.Z")

    def test_version_cuda(self):
        cuda = fvdb.version.cuda
        self.assertIsInstance(cuda, str)
        self.assertTrue(len(cuda) > 0, "cuda version string is empty")
        self.assertRegex(cuda, r"^\d+\.\d+$", f"cuda version {cuda!r} does not match X.Y")

    def test_version_torch(self):
        tv = fvdb.version.torch
        self.assertIsInstance(tv, str)
        self.assertTrue(len(tv) > 0, "torch version string is empty")

    def test_version_git(self):
        g = fvdb.version.git
        self.assertIsInstance(g, str)
        self.assertTrue(len(g) > 0, "git version string is empty")

    def test_version_invalid_attr(self):
        with self.assertRaises(AttributeError):
            _ = fvdb.version.nonexistent_attribute


if __name__ == "__main__":
    unittest.main()
