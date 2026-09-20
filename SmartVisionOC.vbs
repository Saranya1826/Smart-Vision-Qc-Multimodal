Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
appDir = scriptDir & "\Smart Vision Qc-Multimodal"

WshShell.CurrentDirectory = appDir
WshShell.Run "pythonw """ & appDir & "\desktop.py""", 0, False
