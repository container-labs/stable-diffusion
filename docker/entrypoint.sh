#!/bin/sh
# The --login ensures the bash configuration is loaded,
# enabling Conda.

echo "running"
conda init bash
source /root/.bashrc

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
# conda init
conda activate stable-app

# Re-enable strict mode:
set -euo pipefail

# exec the final command:
#exec python3 main.py

tail -f /dev/null
