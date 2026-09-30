#!/bin/bash
set -euo pipefail
cd /app
patch -p1 < /solution/fix.patch
printf '# Fix notes\n\nSee /solution/fix.patch (reference fix).\n' > /app/FIX_NOTES.md
