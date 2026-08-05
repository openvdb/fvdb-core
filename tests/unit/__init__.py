# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import functools

from fvdb.utils.tests import set_testing_git_tag
from parameterized import parameterized

set_testing_git_tag("59e48d3daa8b8fb55a30fd3d7553fc7fa773ab07")


# Hack parameterized to use the function name and the expand parameters as the test name
expand_tests = functools.partial(
    parameterized.expand,
    name_func=lambda f, n, p: f'{f.__name__}_{parameterized.to_safe_name("_".join(str(x) for x in p.args))}',
)
