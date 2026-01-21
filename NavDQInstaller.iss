; -------------------------------------------
; NavDrill Quality Installer
; Folder: C:\Users\Bobby\Desktop\PyProj\Projects\Nav_DrillLog_Conv\NavDQ_v1.0
; -------------------------------------------

[Setup]
AppName=NavDrill Quality
AppVersion=1.0
DefaultDirName={pf}\NavDrill Quality
DefaultGroupName=NavDrill Quality
UninstallDisplayIcon={app}\NavDQ.exe
OutputBaseFilename=NavDQInstaller
Compression=lzma
SolidCompression=yes
SetupIconFile=C:\Users\Bobby\Desktop\PyProj\Projects\Nav_DrillLog_Conv\NavDQ_v1.0\app_icon.ico
DisableStartupPrompt=yes

[Files]
; --- Main Executable ---
Source: "C:\Users\Bobby\Desktop\PyProj\Projects\Nav_DrillLog_Conv\NavDQ_v1.0\dist\NavDrill Quality.exe"; DestDir: "{app}"; Flags: ignoreversion

; --- Splash Screen (optional) ---
Source: "C:\Users\Bobby\Desktop\PyProj\Projects\Nav_DrillLog_Conv\NavDQ_v1.0\splash_screen.png"; DestDir: "{app}"; Flags: ignoreversion

; --- App icon (so it’s available inside app folder if needed) ---
Source: "C:\Users\Bobby\Desktop\PyProj\Projects\Nav_DrillLog_Conv\NavDQ_v1.0\app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu shortcut
Name: "{group}\NavDrill Quality"; Filename: "{app}\NavDrill Quality.exe"
; Desktop shortcut
Name: "{commondesktop}\NavDrill Quality"; Filename: "{app}\NavDrill Quality"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
