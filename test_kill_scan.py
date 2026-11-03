import os
import subprocess

my_pid = os.getpid()
ppid = os.getppid()
print(f"my_pid: {my_pid}, ppid: {ppid}")

cmd = [
    "powershell.exe", "-NoProfile", "-Command",
    "Get-CimInstance Win32_Process -Filter \"Name like '%python%' or Name like '%py%'\" | ForEach-Object { \"$($_.ProcessId)::$($_.Name)::$($_.CommandLine)\" }"
]
out = subprocess.check_output(cmd, text=True, errors="ignore")
for line in out.strip().splitlines():
    if "::" not in line:
        continue
    parts = line.split("::", 2)
    pid_str, name = parts[0], parts[1]
    cmdline = parts[2] if len(parts) > 2 else ""
    pid_str = pid_str.strip()
    if not pid_str.isdigit():
        continue
    pid = int(pid_str)
    print(f"FOUND: pid={pid}, name={name}, is_my_pid={pid == my_pid}, is_ppid={pid == ppid}")
    if "app.py" in cmdline or "paddleocr_server.py" in cmdline or "test_kill" in cmdline:
        print(f"  -> WOULD KILL: pid={pid}, cmdline={cmdline}")

