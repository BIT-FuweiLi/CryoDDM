"""Console entry point for the existing Qt application."""

import runpy


def main():
    runpy.run_module("main", run_name="__main__", alter_sys=True)
