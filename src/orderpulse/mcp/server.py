"""Standardized MCP (Model Context Protocol) Tool Server for Read-Only SQL Operations.

Exposes JSON-Schema typed tools ``get_order_status``,
``get_customer_refund_history``, and ``execute_readonly_sql_audit`` both for
in-process multi-agent orchestration and standalone MCP FastMCP transport.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from orderpulse.database.repository import (
    OperationalOrderRepository,
    ReadOnlySQLGuardrailError,
)


class _OrderStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(..., min_length=4, max_length=32)


class _CustomerHistoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str = Field(..., min_length=4, max_length=32)


class _ReadonlySQLArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=6, max_length=1500)


class OrderPulseMCPServer:
    """MCP-compliant tool dispatcher backed by OperationalOrderRepository."""

    def __init__(self, repository: Optional[OperationalOrderRepository] = None) -> None:
        self.repository = repository or OperationalOrderRepository()
        self._tool_specs: list[dict[str, Any]] = [
            {
                "name": "get_order_status",
                "description": (
                    "Read-only lookup of delivery order status, SLA delay minutes, "
                    "monetary amount, and missing-item metadata by order_id."
                ),
                "inputSchema": _OrderStatusArgs.model_json_schema(),
            },
            {
                "name": "get_customer_refund_history",
                "description": (
                    "Read-only check of customer loyalty tier and 30-day refund frequency "
                    "for abuse/fraud risk evaluation."
                ),
                "inputSchema": _CustomerHistoryArgs.model_json_schema(),
            },
            {
                "name": "execute_readonly_sql_audit",
                "description": (
                    "Executes a validated SELECT/WITH read-only SQL query against the "
                    "operational orders database (mutations strictly blocked)."
                ),
                "inputSchema": _ReadonlySQLArgs.model_json_schema(),
            },
        ]
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "get_order_status": self._handle_get_order_status,
            "get_customer_refund_history": self._handle_get_customer_refund_history,
            "execute_readonly_sql_audit": self._handle_execute_readonly_sql_audit,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP-formatted tool definitions with JSON Schemas."""
        return list(self._tool_specs)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an MCP tool invocation with strict Pydantic argument validation."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            return {
                "isError": True,
                "tool": tool_name,
                "error": f"Unknown MCP tool '{tool_name}'.",
            }
        try:
            return handler(arguments)
        except ReadOnlySQLGuardrailError as exc:
            return {
                "isError": True,
                "tool": tool_name,
                "error": f"SQL_READONLY_GUARDRAIL: {exc}",
            }
        except Exception as exc:
            return {
                "isError": True,
                "tool": tool_name,
                "error": str(exc),
            }

    def _handle_get_order_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = _OrderStatusArgs.model_validate(arguments)
        record = self.repository.get_order_by_id(args.order_id)
        if record is None:
            return {
                "isError": False,
                "tool": "get_order_status",
                "found": False,
                "order_id": args.order_id.upper(),
                "record": None,
            }
        return {
            "isError": False,
            "tool": "get_order_status",
            "found": True,
            "order_id": record.order_id,
            "record": record.model_dump(),
        }

    def _handle_get_customer_refund_history(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = _CustomerHistoryArgs.model_validate(arguments)
        history = self.repository.get_customer_refund_history(args.customer_id)
        return {
            "isError": False,
            "tool": "get_customer_refund_history",
            **history,
        }

    def _handle_execute_readonly_sql_audit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = _ReadonlySQLArgs.model_validate(arguments)
        rows = self.repository.execute_readonly_sql_audit(args.query)
        return {
            "isError": False,
            "tool": "execute_readonly_sql_audit",
            "row_count": len(rows),
            "rows": rows,
        }


def create_fastmcp_server(mcp_server: Optional[OrderPulseMCPServer] = None) -> Any:
    """Optional binding to the official ``mcp`` FastMCP SDK for stdio/SSE serving."""
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

    dispatcher = mcp_server or OrderPulseMCPServer()
    app = FastMCP("orderpulse-operational-mcp")

    @app.tool()
    def get_order_status(order_id: str) -> dict[str, Any]:
        """Lookup operational status and SLA metrics for an order ID."""
        return dispatcher.call_tool("get_order_status", {"order_id": order_id})

    @app.tool()
    def get_customer_refund_history(customer_id: str) -> dict[str, Any]:
        """Lookup customer loyalty tier and 30-day refund history."""
        return dispatcher.call_tool(
            "get_customer_refund_history", {"customer_id": customer_id}
        )

    @app.tool()
    def execute_readonly_sql_audit(query: str) -> dict[str, Any]:
        """Execute a strictly validated SELECT/WITH read-only SQL query."""
        return dispatcher.call_tool("execute_readonly_sql_audit", {"query": query})

    return app
