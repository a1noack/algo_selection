#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='algo-selector',
      version='0.0.1',
      install_requires=['gym>=0.2.3',
                        'pandas',
                        'cfg_load'],
      packages=find_packages(),
      )
