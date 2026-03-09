#!/usr/bin/env python3
"""Convenience entrypoint for simple controls baseline."""
import runpy

if __name__ == '__main__':
    runpy.run_path('double_pendulum_simple_controls_annotated.py', run_name='__main__')
