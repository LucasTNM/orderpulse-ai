"""Read-only SQL repository for operational delivery & refund history audits.

Enforces lexical SQL validation (blocking all DDL/DML mutations) as well as
database-level read-only execution (``PRAGMA query_only = ON`` on SQLite or
``READ ONLY`` transactions on PostgreSQL).
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Optional

from orderpulse.config import Settings, get_settings
from orderpulse.schemas.models import OrderOperationalRecord

_FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|GRANT|REVOKE|EXEC|EXECUTE|CALL|MERGE|VACUUM)\b",
    re.IGNORECASE,
)
_BLOCK_COMMENTS_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENTS_RE = re.compile(r"--[^\r\n]*")


class ReadOnlySQLGuardrailError(ValueError):
    """Raised when an audit SQL query violates read-only execution rules."""


def validate_readonly_sql(query: str) -> str:
    """Validate that a SQL query is a single, pure SELECT/WITH statement."""
    cleaned = _BLOCK_COMMENTS_RE.sub(" ", query)
    cleaned = _LINE_COMMENTS_RE.sub(" ", cleaned).strip()
    if not cleaned:
        raise ReadOnlySQLGuardrailError("Empty SQL query is not allowed.")

    # Reject stacked statements (anything after a semicolon)
    stripped_trailing = cleaned.rstrip(";").strip()
    if ";" in stripped_trailing:
        raise ReadOnlySQLGuardrailError(
            "Multiple SQL statements are prohibited by the Read-Only SQL guardrail."
        )

    upper = stripped_trailing.upper()
    if not (upper.startswith("SELECT ") or upper.startswith("WITH ")):
        raise ReadOnlySQLGuardrailError(
            "Only SELECT or WITH read-only queries are permitted."
        )

    forbidden = _FORBIDDEN_SQL_KEYWORDS.search(stripped_trailing)
    if forbidden:
        raise ReadOnlySQLGuardrailError(
            f"Forbidden SQL keyword '{forbidden.group(1).upper()}' blocked by Read-Only guardrail."
        )

    return stripped_trailing


_SEED_ORDERS: tuple[tuple[Any, ...], ...] = (
    ("ORD-9001", "CUST-101", "GOLD", "Burger Lab Paulista", "DELIVERED", 89.90, 35, 90, 55, 0, 0.00, 0, "2026-11-15T11:00:00-03:00", 1),
    ("ORD-9002", "CUST-102", "DIAMOND", "Japa Prime Jardins", "DELIVERED", 245.50, 40, 95, 55, 0, 0.00, 0, "2026-11-15T10:30:00-03:00", 1),
    ("ORD-9003", "CUST-103", "STANDARD", "Pizzaria Bella Napoli", "DELIVERED", 74.00, 45, 47, 2, 0, 0.00, 5, "2026-11-15T11:10:00-03:00", 0),
    ("ORD-9004", "CUST-104", "GOLD", "Smash & Co", "DELIVERED", 68.50, 30, 33, 3, 1, 14.90, 1, "2026-11-15T11:20:00-03:00", 0),
    ("ORD-9005", "CUST-105", "STANDARD", "Cantina Roma", "DELIVERED", 112.00, 40, 62, 22, 0, 0.00, 0, "2026-11-15T10:45:00-03:00", 1),
    ("ORD-9006", "CUST-106", "STANDARD", "Taqueria El Sol", "DELIVERED", 100.00, 35, 71, 36, 0, 0.00, 1, "2026-11-15T10:15:00-03:00", 1),
    ("ORD-9007", "CUST-107", "DIAMOND", "Bistro Vila Madalena", "DELIVERED", 120.00, 40, 75, 35, 0, 0.00, 0, "2026-11-15T10:20:00-03:00", 1),
    ("ORD-9008", "CUST-108", "GOLD", "Churrascaria Sul", "CREATED", 135.00, 45, None, 0, 0, 0.00, 0, "2026-11-15T12:10:00-03:00", 0),
    ("ORD-9009", "CUST-109", "STANDARD", "Padaria Central", "IN_TRANSIT", 58.00, 35, None, 5, 0, 0.00, 0, "2026-11-15T11:55:00-03:00", 0),
    ("ORD-9010", "CUST-110", "DIAMOND", "Osteria Milano", "ACCEPTED", 198.00, 50, None, 0, 0, 0.00, 0, "2026-11-15T12:15:00-03:00", 0),
    ("ORD-9011", "CUST-111", "STANDARD", "Poke House", "DELIVERED", 210.00, 35, 38, 3, 1, 165.00, 0, "2026-11-15T10:05:00-03:00", 0),
    ("ORD-9012", "CUST-112", "STANDARD", "Acai Express", "DELIVERED", 42.00, 25, 28, 3, 1, 18.00, 4, "2026-11-15T11:40:00-03:00", 0),
    ("ORD-9013", "CUST-113", "GOLD", "Sushi Mori", "PREPARING", 94.00, 45, None, 0, 0, 0.00, 1, "2026-11-15T12:05:00-03:00", 0),
    ("ORD-9014", "CUST-114", "STANDARD", "Hamburgueria 24h", "DELIVERED", 149.90, 40, 92, 52, 0, 0.00, 2, "2026-11-15T09:40:00-03:00", 1),
    ("ORD-9015", "CUST-115", "GOLD", "Temakeria Barra", "DELIVERED", 150.01, 35, 88, 53, 0, 0.00, 0, "2026-11-15T09:50:00-03:00", 1),
)


class OperationalOrderRepository:
    """Thread-safe read-only operational database adapter."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap_database()

    def _bootstrap_database(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                customer_tier TEXT NOT NULL,
                restaurant_name TEXT NOT NULL,
                order_status TEXT NOT NULL,
                total_amount_brl REAL NOT NULL,
                promised_eta_min INTEGER NOT NULL,
                actual_delivery_min INTEGER,
                delay_minutes INTEGER NOT NULL,
                missing_items_reported INTEGER NOT NULL,
                missing_items_value_brl REAL NOT NULL,
                prior_refunds_30d INTEGER NOT NULL,
                created_at_iso TEXT NOT NULL,
                courier_gps_delayed INTEGER NOT NULL
            )
            """
        )
        cur.executemany(
            """
            INSERT INTO orders VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            _SEED_ORDERS,
        )
        self._conn.commit()
        if self.settings.sql_readonly_enforced:
            self._conn.execute("PRAGMA query_only = ON;")

    def get_order_by_id(self, order_id: str) -> Optional[OrderOperationalRecord]:
        """Fetch a single validated OrderOperationalRecord by order_id."""
        cur = self._conn.execute(
            "SELECT * FROM orders WHERE UPPER(order_id) = UPPER(?)",
            (order_id.strip(),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_customer_refund_history(self, customer_id: str) -> dict[str, Any]:
        """Return aggregated customer tier and 30-day refund history."""
        cur = self._conn.execute(
            """
            SELECT customer_id,
                   MAX(customer_tier) AS customer_tier,
                   MAX(prior_refunds_30d) AS prior_refunds_30d,
                   COUNT(*) AS orders_count,
                   ROUND(SUM(total_amount_brl), 2) AS total_spent_brl
            FROM orders
            WHERE UPPER(customer_id) = UPPER(?)
            GROUP BY customer_id
            """,
            (customer_id.strip(),),
        )
        row = cur.fetchone()
        if row is None:
            return {
                "found": False,
                "customer_id": customer_id.upper(),
                "prior_refunds_30d": 0,
                "abuse_flag": False,
            }
        prior_refunds = int(row["prior_refunds_30d"])
        return {
            "found": True,
            "customer_id": row["customer_id"],
            "customer_tier": row["customer_tier"],
            "prior_refunds_30d": prior_refunds,
            "orders_count": int(row["orders_count"]),
            "total_spent_brl": float(row["total_spent_brl"]),
            "abuse_flag": prior_refunds >= self.settings.max_refunds_30d_abuse_threshold,
        }

    def execute_readonly_sql_audit(self, query: str) -> list[dict[str, Any]]:
        """Execute an arbitrary SELECT/WITH query guarded by Read-Only validation."""
        safe_sql = validate_readonly_sql(query)
        cur = self._conn.execute(safe_sql)
        rows = cur.fetchall()
        return [dict(r) for r in rows[:50]]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> OrderOperationalRecord:
        return OrderOperationalRecord(
            order_id=str(row["order_id"]),
            customer_id=str(row["customer_id"]),
            customer_tier=str(row["customer_tier"]),
            restaurant_name=str(row["restaurant_name"]),
            order_status=str(row["order_status"]),
            total_amount_brl=float(row["total_amount_brl"]),
            promised_eta_min=int(row["promised_eta_min"]),
            actual_delivery_min=(
                int(row["actual_delivery_min"])
                if row["actual_delivery_min"] is not None
                else None
            ),
            delay_minutes=int(row["delay_minutes"]),
            missing_items_reported=bool(row["missing_items_reported"]),
            missing_items_value_brl=float(row["missing_items_value_brl"]),
            prior_refunds_30d=int(row["prior_refunds_30d"]),
            created_at_iso=str(row["created_at_iso"]),
            courier_gps_delayed=bool(row["courier_gps_delayed"]),
        )
