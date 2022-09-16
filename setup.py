from setuptools import setup
setup(name='pymeasfrf',
version='0.1',
description='1st version of the measurement toolbox',
url='#',
author='Nuopel',
author_email='samu.dupont@laposte.net',
license='GNU General Public License v3.0',
packages=['pymeasfrf'],
  install_requires=[
      'sounddevice',
      'numpy',
      'scipy',
      'pathlib',
      'matplotlib'
  ],
zip_safe=False)

