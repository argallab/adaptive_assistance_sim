## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['envs'],
    package_dir={'': 'src'})


setup(**setup_args)
