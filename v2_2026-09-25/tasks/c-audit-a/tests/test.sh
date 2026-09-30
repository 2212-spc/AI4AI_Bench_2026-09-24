#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/verify_c.py || echo 0 > /logs/verifier/reward.txt
