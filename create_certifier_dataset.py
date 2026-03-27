"""Compatibility entrypoint for offline vote dataset generation.

Canonical implementation lives in `nncert.cli.create_votes_dataset`.
"""

from nncert.cli.create_votes_dataset import main


if __name__ == "__main__":
    main()
