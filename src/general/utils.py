import os
import sys
import src.general.global_variables as gv

sys.path.append(gv.PROJECT_PATH)

def cc_path(file_path: str) -> str:
    """Create absolute path."""
    return os.path.join(gv.PROJECT_PATH, file_path)