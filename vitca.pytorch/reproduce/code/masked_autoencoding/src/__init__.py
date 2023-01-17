from pathlib import Path
import sys
if str(Path(__file__).parent.parent.parent.resolve()) not in sys.path:
	sys.path.insert(1, str(Path(__file__).parent.parent.parent.resolve()))