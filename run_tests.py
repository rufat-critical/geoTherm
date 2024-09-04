import subprocess

def run_tests():
    try:
        # Run pytest with a timeout of 300 seconds (5 minutes)
        result = subprocess.run(['pytest', '--tb=short'], timeout=300)
        
        # Check the return code to determine if tests passed or failed
        if result.returncode != 0:
            print("Some tests failed. Aborting push.")
            return False
        else:
            print("All tests passed!")
            return True
    except subprocess.TimeoutExpired:
        print("Tests took too long to run. Aborting push.")
        return False
    except Exception as e:
        print(f"An error occurred while running tests: {e}")
        return False

if __name__ == "__main__":
    if not run_tests():
        exit(1)
