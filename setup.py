from setuptools import setup, find_packages
import os
from setuptools.command.develop import develop


# Custom command class to run the hook setup script
class DevelopWithHooks(develop):
    def run(self):
        os.system('python setup_hooks.py')
        super().run()

setup(
    name="geoTherm",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    author="Rufat Kulakhmetov",
    author_email="rufat@criticalenergy.co",
    description="Lumped Parameter thermo fluid modeling",
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    package_data={},
    install_requires=[
        'CoolProp',
        'scipy==1.14',
        'matplotlib',
        'numpy==2.0',
        'rich',
        'pint',
        'pytest',
        'pyyed',
        'plantuml',
        'pandas==2.2.2',
    ],
    cmdclass={
        'develop': DevelopWithHooks,
    }
)
