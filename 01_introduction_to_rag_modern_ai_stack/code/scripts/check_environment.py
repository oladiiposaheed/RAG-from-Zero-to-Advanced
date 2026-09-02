'''Verifies Python version and that all key libraries can be imported.
'''

import sys
import importlib

# Minimum Python version: 3.9
REQUIRED_PYTHON = (3, 9)

LIBS = [
    'langchain', 
    'langchain_openai',
    'langchain_google_genai',
    'langchain_anthropic',
    'langgraph',
    'langsmith',
    'dotenv', 
    'pydantic']


def check_python():
    '''
    Check if the current Python version is at least the required minimum.
    Returns True if sufficient, False otherwise.
    '''
    
    # Get major and minor version
    major, minor  = sys.version_info[:2]
    print(sys.version_info)
    
    # Check version
    if (major, minor) < REQUIRED_PYTHON:
        print(f'Python {major}.{minor} detected. Need >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}')
        return False   # Indicate failure
    
    print(f'Python version: {major}.{minor}')
    return True


def check_libs():
    '''
    Try to import each library. Returns True if successful.
    '''
    
    all_ok = True
    
    # Loop through libraries
    for lib in LIBS:
        try:
            # Import module
            importlib.import_module(lib)
            print(f'{lib} imported successfully.')
        
        except ImportError as e:
            print(f'{lib} import failed: {e}')
            all_ok = False
            
    return all_ok


if __name__ == '__main__':
    print('Checking environment...\n')
    
    # Check python version and lib import and store results
    py_ok = check_python()
    libs_ok = check_libs()
    
    # Verify all passed
    if py_ok and libs_ok:
        print('\n Environment is ready')
    else:
        print('\n Please fix the issues above before proceeding.')