@echo off
rem Started through explorer.exe so it runs as the normal user and survives the Claude Code session.
cd /d C:\Users\sadan\projects\next-gen-lab\colibri
start "" /min C:\Users\sadan\projects\ncm-lab\.venv\Scripts\pythonw.exe run_colibri.py
