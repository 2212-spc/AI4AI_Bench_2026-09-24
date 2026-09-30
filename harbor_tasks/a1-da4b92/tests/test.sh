#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/verify_a1.py || echo 0 > /logs/verifier/reward.txt
