import signal
import subprocess
import sys
import time
import os
from pathlib import Path


ROOT       = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"


def spawn(name: str, args: list[str], env: dict | None = None) -> subprocess.Popen:
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