Installation on Windows 10:

download docker from https://docs.docker.com/desktop/windows/install/
enable Windows Subsytem for Linux: 
-   open windows powershell as administrator and run: dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
-   run dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
-   restart computer
-   download linux kernel update package: https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi and double click to run
-   open powershell and run wsl --set-default-version 2
open microsoft store and select linux distribution of choice, select get

see more here

