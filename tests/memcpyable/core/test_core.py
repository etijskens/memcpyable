#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for C++ module memcpyable.core.
"""

import sys
sys.path.insert(0,'.')

import numpy as np

import memcpyable.core


def test():
    ok = memcpyable.core.test_memcpy_able()
    print(f"{ok=}")
    assert ok
    
    ok = memcpyable.core.test_traits()


#===============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
#===============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test

    print(f"__main__ running {the_test_you_want_to_debug} ...")
    the_test_you_want_to_debug()
    print('-*# finished #*-')
#===============================================================================
