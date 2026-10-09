#!/usr/bin/env python3
"""Точка входа: python run.py [--headless N]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beamng_lite.main import main
sys.exit(main())
