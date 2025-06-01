import subprocess
import sys
from typing import List
from .config import REQUIRED_PACKAGES

def check_dependencies() -> bool:
    """Check if all required packages are installed."""
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing dependencies:", ", ".join(missing_packages))
        return False
    return True

def install_package(package: str) -> bool:
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, 
                             "--no-cache-dir", "--verbose", "--force-reinstall", "--prefer-binary"])
        return True
    except subprocess.CalledProcessError:
        return False

def print_status(message: str, status: str = "info"):
    """Print a status message with appropriate formatting."""
    status_symbols = {
        "info": "ℹ️",
        "success": "✓",
        "error": "❌",
        "warning": "⚠️"
    }
    print(f"{status_symbols.get(status, '')} {message}") 