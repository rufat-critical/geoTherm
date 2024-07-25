from setuptools import setup, find_packages


setup(
    name="geoTherm",
    version="1.0.0",
    packages = find_packages(),
    include_package_data=True,
    author="Rufat Kulakhmetov",
    author_email="rufat.kulakhmetov@gmail.com",
    description="Geothermal Thermodynamics Model",
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    package_data={},
    install_requires=[
        'CoolProp',
        'scipy',
        'matplotlib',
        'numpy',
        'rich',
        'pint',
        'pytest'
    ],
)