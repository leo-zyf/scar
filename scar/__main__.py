"""The module and installed console script share the same lazy dispatcher."""
from scar.entrypoint import main

if __name__ == '__main__':
    raise SystemExit(main())
