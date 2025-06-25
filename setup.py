from setuptools import setup, find_packages
import os
import sys

def create_git_hook():
    """
    Create a pre-commit git hook to run tests before committing code.
    """
    git_hooks_dir = os.path.join(os.getcwd(), '.git', 'hooks')
    pre_commit_path = os.path.join(git_hooks_dir, 'pre-commit')
    
    # Check if .git directory exists
    if not os.path.exists(git_hooks_dir):
        print("Not a git repository. Skipping git hook installation.")
        return

    # Get the path of the Python interpreter used for pip installation
    python_executable = sys.executable

    hook_content = f"""#!/bin/bash
"{python_executable}" -m pytest tests
if [ $? -ne 0 ]; then
  echo "Tests failed. Aborting commit."
  exit 1
fi
"""

    # Create the git hooks directory if it doesn't exist
    os.makedirs(git_hooks_dir, exist_ok=True)

    # Write the pre-commit hook
    with open(pre_commit_path, 'w') as hook_file:
        hook_file.write(hook_content)

    # Make the hook executable
    os.chmod(pre_commit_path, 0o775)
    print("Pre-commit hook created at .git/hooks/pre-commit")

create_git_hook()

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
        'pyyaml',
        'tabulate',
    ],
)
