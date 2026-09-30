#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/verify_s.py || echo 0 > /logs/verifier/reward.txt
