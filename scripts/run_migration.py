"""Quick script to run the TimescaleDB schema migration."""
import sys
import os
import logging

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from src.db.db import run_migration, close_pool

if __name__ == "__main__":
    try:
        run_migration()
        print("\n✅ Schema migration completed successfully!")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        close_pool()
