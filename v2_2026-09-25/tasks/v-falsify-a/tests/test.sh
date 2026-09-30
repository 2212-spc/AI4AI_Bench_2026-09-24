#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/verify_v.py || echo 0 > /logs/verifier/reward.txt
