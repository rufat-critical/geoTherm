import os
import platform
import stat


# This line will be dynamically updated by the custom install command
python_path = r"C:\Users\rufat\anaconda3\envs\geotherm-stable\python"

def create_pre_push_hook():
    # Detect the operating system
    system_type = platform.system()

    # Determine the project's root directory
    project_root = os.getcwd()

    if system_type == "Windows":
        hook_content = f"""@echo off
REM Navigate to the project's root directory
cd /d {project_root}
REM Run the Python test script using the detected Python executable
"{python_path}" run_tests.py
REM Check if the test script failed
IF %ERRORLEVEL% NEQ 0 (
    echo Push aborted due to failed tests.
    exit /b 1
)
"""
        hook_filename = 'pre-push.bat'
    else:
        hook_content = f"""#!/bin/bash
# Navigate to the project's root directory
cd "{project_root}"
# Run the Python test script using the detected Python executable
"{python_path}" run_tests.py
# Check if the test script failed
if [ $? -ne 0 ]; then
    echo "Push aborted due to failed tests."
    exit 1
fi
"""
        hook_filename = 'pre-push'

    # Define the hook path
    git_hooks_dir = os.path.join(project_root, '.git', 'hooks')
    pre_push_hook_path = os.path.join(git_hooks_dir, hook_filename)

    # Create the hooks directory if it doesn't exist
    os.makedirs(git_hooks_dir, exist_ok=True)

    # Write the hook file
    with open(pre_push_hook_path, 'w') as hook_file:
        hook_file.write(hook_content)

    # Make the hook executable on Unix-like systems
    if system_type != "Windows":
        os.chmod(pre_push_hook_path, os.stat(pre_push_hook_path).st_mode | stat.S_IEXEC)

    print(f"Git pre-push hook created successfully for {system_type}.")

if __name__ == "__main__":
    create_pre_push_hook()
