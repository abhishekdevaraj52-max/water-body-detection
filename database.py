"""
database.py — SQLite helper module for Water Body Analysis.

All functions use parameterised queries; no raw string interpolation.
Relative image paths (e.g. outputs/original/analysis_0001.png) are stored
in the DB so the project is portable across machines.
"""

import os
import sqlite3
import datetime
from contextlib import contextmanager

import config  # DB_PATH, DB_DIR already guaranteed to exist


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """Yield a SQLite connection with row_factory set and auto-close it."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL") # safe for concurrent readers
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_database() -> None:
    """Create the water_analysis table if it does not already exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS water_analysis (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id          TEXT    UNIQUE NOT NULL,
        image_name           TEXT,
        original_image_path  TEXT    NOT NULL,
        predicted_mask_path  TEXT    NOT NULL,
        location             TEXT,
        latitude             REAL,
        longitude            REAL,
        water_area_m2        REAL,
        water_area_ha        REAL,
        water_area_km2       REAL,
        water_pixels         INTEGER,
        water_percentage     REAL,
        created_at           TEXT
    );
    """
    with get_connection() as conn:
        conn.execute(ddl)


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def generate_analysis_id() -> str:
    """
    Return the next sequential analysis ID, e.g. 'analysis_0001'.
    Reads the current MAX id from the DB so IDs never collide even after
    app restarts.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT analysis_id FROM water_analysis ORDER BY id DESC LIMIT 1"
        ).fetchone()

    if row is None:
        return "analysis_0001"

    last = row["analysis_id"]          # e.g. 'analysis_0042'
    try:
        num = int(last.split("_")[1])
    except (IndexError, ValueError):
        num = 0
    return f"analysis_{num + 1:04d}"


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def insert_analysis(
    analysis_id: str,
    image_name: str,
    original_image_path: str,
    predicted_mask_path: str,
    location: str = None,
    latitude: float = None,
    longitude: float = None,
    water_area_m2: float = None,
    water_area_ha: float = None,
    water_area_km2: float = None,
    water_pixels: int = None,
    water_percentage: float = None,
) -> int:
    """
    Insert one analysis record and return its auto-incremented row id.
    Paths stored as relative strings (no absolute Windows paths).
    """
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = """
    INSERT INTO water_analysis (
        analysis_id, image_name,
        original_image_path, predicted_mask_path,
        location, latitude, longitude,
        water_area_m2, water_area_ha, water_area_km2,
        water_pixels, water_percentage,
        created_at
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    params = (
        analysis_id, image_name,
        original_image_path, predicted_mask_path,
        location, latitude, longitude,
        water_area_m2, water_area_ha, water_area_km2,
        water_pixels, water_percentage,
        created_at,
    )
    with get_connection() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_all_analyses() -> list[dict]:
    """Return all analyses ordered newest-first as a list of dicts."""
    sql = """
    SELECT id, analysis_id, image_name, location,
           water_area_m2, water_area_ha, water_area_km2,
           water_pixels, water_percentage, created_at
    FROM water_analysis
    ORDER BY id DESC
    """
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id: str) -> dict | None:
    """Return a single analysis record as a dict, or None if not found."""
    sql = "SELECT * FROM water_analysis WHERE analysis_id = ?"
    with get_connection() as conn:
        row = conn.execute(sql, (analysis_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_database()
    print(f"[OK] Database initialised at: {config.DB_PATH}")
    test_id = generate_analysis_id()
    print(f"[OK] Next analysis ID would be: {test_id}")
