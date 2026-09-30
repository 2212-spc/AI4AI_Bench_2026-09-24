#!/bin/bash
mkdir -p /logs/verifier
HARBOR_CONTAINER=1 python3 /tests/verify_o.py || echo 0 > /logs/verifier/reward.txt
