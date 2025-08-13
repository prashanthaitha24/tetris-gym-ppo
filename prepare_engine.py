from pathlib import Path

def main():
    try:
        from git import Repo
    except Exception:
        print("Install GitPython first: pip install gitpython")
        return
    target = Path(__file__).parent / "engines" / "nuno_faria"
    if target.exists() and any(target.iterdir()):
        print(f"[ok] Engine already present: {target}")
        return
    Repo.clone_from("https://github.com/nuno-faria/tetris-ai.git", target)
    print("[ok] Cloned engine to", target)

if __name__ == "__main__":
    main()
