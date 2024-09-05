# geoTherm

Geothermal Thermodynamic Model

Installation Instructions:
1) Set up a conda virtual environment
2) Install assimulo conda install conda-forge::assimulo
3) Clone the repo
4) Run pip install . from the main folder
5) run pytest and make sure test cases pass
6) You're all set up!

*** For Developers
The setup.py script sets up a git pre-commit hook that runs everytime a commit is made
The pre-commit hook runs pytest to make sure test cases pass. If the commit fails then
check the test cases and fix issues before committing. 