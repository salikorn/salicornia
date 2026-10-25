# salikorn — extend WDDM TDR so long GPU compute packets don't reset the device.
# TdrDelay: seconds a GPU packet may run before recovery kicks in (default 2s).
# TdrDdiDelay: seconds a DDI call may take (default 5s).
# Recovery stays ENABLED (TdrLevel untouched) — a real hang still recovers.
# Takes effect after reboot (or driver restart).
$newItem = New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' `
    -Name 'TdrDelay' -PropertyType DWord -Value 10 -Force
$newItem2 = New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' `
    -Name 'TdrDdiDelay' -PropertyType DWord -Value 15 -Force
Write-Output ("TdrDelay=" + $newItem.TdrDelay + " TdrDdiDelay=" + $newItem2.TdrDdiDelay)
