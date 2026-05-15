#!/usr/bin/env python3
import os
import subprocess
import sys

if __name__ == '__main__':
    script = os.path.join(os.path.dirname(__file__), 'run.sh')
    raise SystemExit(subprocess.call(['bash', script] + sys.argv[1:]))
