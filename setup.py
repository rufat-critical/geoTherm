from setuptools import setup, find_packages
import os
from setuptools.command.develop import develop

# Custom command to detect Python interpreter and update setup_hooks.py
class CustomDevelopCommand(develop):
    def run(self):
        # Detect the current Python interpreter path
        python_path = os.path.abspath(os.path.realpath(os.path.join(os.__file__, '..', '..', 'python')))
        
        # Update the setup_hooks.py with the detected Python path
        update_setup_hooks(python_path)

        # Run the standard setuptools develop command
        super().run()

def update_setup_hooks(python_path):
    """Updates setup_hooks.py with the detected Python interpreter path."""
    setup_hooks_path = os.path.join(os.getcwd(), 'setup_hooks.py')
    
    # Read the current content of setup_hooks.py
    with open(setup_hooks_path, 'r') as file:
        setup_hooks_content = file.readlines()

    # Replace the placeholder or previous Python path with the new one
    updated_content = []
    for line in setup_hooks_content:
        if line.strip().startswith('python_path ='):
            updated_content.append(f'python_path = r"{python_path}"\n')
        else:
            updated_content.append(line)

    # Write the updated content back to setup_hooks.py
    with open(setup_hooks_path, 'w') as file:
        file.writelines(updated_content)
    
    print(f"Updated setup_hooks.py with Python path: {python_path}")

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
        'develop': CustomDevelopCommand,
    }
)
