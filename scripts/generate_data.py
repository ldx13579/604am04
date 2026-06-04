"""Generate 1M offline transitions using random policy and store in PostgreSQL."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import Base, engine
from backend.data.generator import data_generator


def main():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

    print("Generating 1,000,000 offline transitions with random policy...")
    print("This may take 3-5 minutes...")

    def progress_callback(progress):
        bar_len = 40
        filled = int(bar_len * progress)
        bar = '=' * filled + '-' * (bar_len - filled)
        print(f'\r[{bar}] {progress*100:.1f}%', end='', flush=True)

    data_generator.generate(callback=progress_callback)
    print(f"\nDone! Generated {data_generator.total_generated:,} transitions.")


if __name__ == "__main__":
    main()
