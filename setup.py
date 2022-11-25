from setuptools import setup
setup(name='pymeasfrf',
version='1',
description='Unlicenced branch main_al of numerous modules, classes, for acoustic measurement from pymeasfrf main ',
url='#',
author='Samuel Dupont',
author_email='samuel.dupont@arteac-lab.fr',
packages=['pymeasfrf'],
  install_requires=[
      'sounddevice',
      'numpy',
      'scipy',
      'pathlib',
      'matplotlib'
  ],
zip_safe=False)

