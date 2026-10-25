"""Overnight driver: start Argus's "Think harder" service (DeepSeek V4 Flash 284B through colibri on 127.0.0.1:8796,
which first unloads the local model), wait for it, run bench_quality.py against it, then stop it. Log: results/colibri-run.log."""
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARGUS = Path(r"C:\Users\sadan\my-agent\argus")
LOG = HERE / "results" / "colibri-run.log"


def log(msg):
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def up(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


log("starting think-harder")
srv = subprocess.Popen([str(ARGUS / "memory-server" / ".venv" / "Scripts" / "python.exe"), str(ARGUS / "think-harder" / "think_harder.py")],
                       cwd=str(ARGUS), stdout=open(HERE / "results" / "think-harder.log", "a", encoding="utf-8"),
                       stderr=subprocess.STDOUT, creationflags=0x08000000)
t = time.time()
while not up(8796) and time.time() - t < 3600 and srv.poll() is None:
    time.sleep(10)
log(f"server {'up' if up(8796) else 'NOT up'} after {time.time() - t:.0f}s (exit {srv.poll()})")
try:
    if up(8796):
        r = subprocess.run([sys.executable, str(HERE / "bench_quality.py"), "--backend", "openai", "--url", "http://127.0.0.1:8796/v1",
                            "--model", "deepseek-v4-colibri", "--name", "deepseek-v4-colibri", "--n", "10", "--timeout", "7200"],
                           cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")
        log("bench exit %s\n%s%s" % (r.returncode, r.stdout[-3000:], r.stderr[-1500:]))
finally:
    subprocess.run(["taskkill", "/PID", str(srv.pid), "/T", "/F"], capture_output=True, creationflags=0x08000000)
    log("think-harder stopped")
