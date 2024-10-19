@echo off

rem auto run the blender script
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --python "%~dp0model_creation.py"

pause