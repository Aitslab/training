Installation on Windows 10:

download docker from https://docs.docker.com/desktop/windows/install/

enable Windows Subsytem for Linux: 
-   open windows powershell as administrator and run: dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
-   run dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
-   restart computer
-   download linux kernel update package: https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi and double click to run
-   open powershell and run wsl --set-default-version 2

double click docker installation file; when prompted, ensure the Enable Install required Windows components for WSL 2 option is selected on the Configuration page.

If your admin account is different to your user account, you must add the user to the docker-users group. Run Computer Management as an administrator and navigate to Local Users and Groups > Groups > docker-users. Right-click to add the user to the group. Log out and log back in for the changes to take effect.

see more info here: https://docs.docker.com/desktop/windows/install/

