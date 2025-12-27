import os

# =========================================================
# .env loader (robust)
# =========================================================

def load_env_robust():
    here = os.path.dirname(os.path.abspath(__file__))
    env_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(here, ".env"),
    ]

    loaded_path = None
    for p in env_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
                loaded_path = p
                break
            except Exception as e:
                print(f"ENV | failed to read {p}: {e}")

    if loaded_path:
        print(f"ENV | .env loaded from: {loaded_path}")
    else:
        print("ENV | .env not found, using process env only")
