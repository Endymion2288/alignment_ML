#!/usr/bin/env python3
"""Reuse immutable WB96 event configuration with an isolated diagnostic component."""
from pathlib import Path
old=Path(__file__).with_name('wb96_athena.py').read_text()
old=old.replace('WB96.ToleranceNavigationAudit','WB99Component.DirectionDoublingAudit').replace('WB96ToleranceNavigationAudit','WB99DirectionDoublingAudit').replace('WB96OfficialControl','WB99OfficialControl')
exec(compile(old,str(Path(__file__).with_name('wb96_athena.py')),'exec'))
