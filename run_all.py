"""
run_all.py
==========
Unified process launcher for the Keep The Town Clean illegal dumping detection platform.

This script starts both backend services with a single command and keeps them running
until the user stops them or one of them exits unexpectedly. It handles cross-platform
process management for macOS, Linux, and Windows, including cleanup of Flask reloader
child processes on Windows.

Services started:
    - Map service:     generates and serves the live Oakland risk dashboard
                       available at http://127.0.0.1:8080/public/oakland_dashboard.html
    - Backend service: Flask API server handling image detection and report endpoints
                       available at http://127.0.0.1:8000/api/health

Usage:
    python run_all.py

Stop:
    Press Ctrl+C. Both services are terminated cleanly before the script exits.

Notes:
    - Flask debug mode and the reloader are disabled intentionally when launched
      from this script to prevent duplicate child processes.
    - Both services persist state via files rather than in-flight requests, so
      no grace period is needed on shutdown.
    - If either service exits on its own, the script immediately stops the other
      and exits with a non-zero return code.
"""
import signal
import subprocess
import sys
import time
import os
from pathlib import Path

# Project root is the directory this file lives in.
# All subprocess commands are run relative to this path.
ROOT       = Path(__file__).resolve().parent

# Used throughout to switch between Windows and Unix process management behavior.
IS_WINDOWS = os.name == "nt"

def spawn(name: str, args: list[str], env: dict | None = None) -> subprocess.Popen:
    """
    Launch a subprocess and return the Popen handle.

    On Windows, CREATE_NEW_PROCESS_GROUP is set so the process can be
    killed as a tree later via taskkill. On Unix this flag is not needed.

    Args:
        name: Human-readable label used in log output.
        args: Full command and argument list passed to Popen.
        env:  Optional environment dictionary. Inherits the current
              environment if None.

    Returns:
        The running Popen instance.
    """
    print(f"Starting {name}: {' '.join(args)}")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    return subprocess.Popen(args, cwd=ROOT, env=env, creationflags=creationflags)

def kill_tree(proc: subprocess.Popen, name: str):
    """Terminate the entire process tree — works around Flask reloader children on Windows."""
    if proc.poll() is not None:
        return
    print(f"Stopping {name}...")
    try:
        if IS_WINDOWS:
            # /T kills the process AND all its children (e.g. Flask reloader)
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.terminate()
    except Exception:
        pass

def main() -> int:
    """
    Entry point. Starts both services, monitors them, and shuts everything
    down cleanly on exit.

    Startup order:
        1. Map service starts first because the frontend depends on the
           dashboard HTML being available before the backend is ready.
        2. A short delay gives the map service time to initialize before
           the backend comes up alongside it.
        3. Each service is checked immediately after launch. If either
           exits within the startup window, the script aborts early.

    Shutdown:
        SIGINT (Ctrl+C) and SIGTERM both trigger the shutdown handler,
        which kills all process trees in reverse startup order and waits
        briefly for the OS to release file handles before the script exits.

    Returns:
        0 on clean exit, 1 if a service failed to start or exited unexpectedly.
    """
    python_exe   = sys.executable
    base_env     = os.environ.copy()
    # Disable Flask debug/reloader when launched from here
    backend_env = base_env.copy()
    backend_env["FLASK_DEBUG"] = "0"
    backend_env["FLASK_ENV"]   = "production"
    backend_args = [python_exe, "backend/server.py"]
    map_args     = [python_exe, "map/src/build_live_dashboard.py", "--serve"]
    processes: list[tuple[str, subprocess.Popen]] = []
    shutting_down = False
    def shutdown(*_):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        # Kill all process trees immediately — no grace period needed since
        # both services handle state via files, not in-flight requests.
        for name, proc in reversed(processes):
            kill_tree(proc, name)
        # Brief wait for OS to clean up handles
        for _, proc in processes:
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        processes.append(("map-service", spawn("map-service", map_args, env=base_env)))
        time.sleep(0.7)
        if processes[-1][1].poll() is not None:
            print("map-service exited early. Check errors above.")
            return 1
        processes.append(("backend-service", spawn("backend-service", backend_args, env=backend_env)))
        if processes[-1][1].poll() is not None:
            print("backend-service exited early. Check errors above.")
            return 1
        print("\nServices running:")
        print("  - Backend API:   http://127.0.0.1:8000/api/health")
        print("  - Map dashboard: http://127.0.0.1:8080/public/oakland_dashboard.html")
        print("\nPress Ctrl+C to stop.\n")
        while not shutting_down:
            for name, proc in processes:
                code = proc.poll()
                if code is not None and not shutting_down:
                    print(f"{name} exited with code {code}; stopping remaining services.")
                    shutdown()
                    break
            time.sleep(0.5)
    finally:
        shutdown()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
