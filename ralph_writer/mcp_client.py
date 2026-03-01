"""MCP (Model Context Protocol) client for Ralph Writer.

Provides a registry that manages connections to MCP servers and translates
their tool schemas into OpenAI-compatible function definitions.  Tools are
namespaced as ``mcp__<server>__<tool>`` to avoid collisions with built-in
tools.

Supported transports:
  - **sse**: Connects to an SSE endpoint (``url``).

The registry runs a background asyncio event loop in a daemon thread so that
SSE connections stay alive between synchronous ``connect`` / ``call_tool`` /
``disconnect_all`` calls made by the orchestrator.

Usage::

    registry = MCPToolRegistry(server_configs)
    registry.connect("snippetts_basic")        # blocking
    defs = registry.list_tool_definitions("snippetts_basic")
    result = registry.call_tool("snippetts_basic", "web_search", {"query": "..."})
    registry.disconnect_all()
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCPServerConfig:
    """Parsed configuration for a single MCP server.

    Attributes:
        name: Logical name used in phase ``tools.mcp`` lists.
        transport: ``"streamable_http"`` (default), ``"sse"``, or ``"auto"``.
        url: The HTTP endpoint URL.
        env: Extra environment variables (reserved for future use).
    """

    name: str
    transport: str = "streamable_http"
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "MCPServerConfig":
        return cls(
            name=name,
            transport=data.get("transport", "streamable_http"),
            url=data.get("url", ""),
            env=data.get("env", {}),
        )


def parse_mcp_configs(
    raw: dict[str, dict[str, Any]],
) -> dict[str, MCPServerConfig]:
    """Parse the ``mcp_servers:`` YAML block into typed configs."""
    return {
        name: MCPServerConfig.from_dict(name, data) for name, data in raw.items()
    }


# ---------------------------------------------------------------------------
# Tool-name helpers
# ---------------------------------------------------------------------------

MCP_TOOL_PREFIX = "mcp__"


def mcp_tool_name(server: str, tool: str) -> str:
    """Build a namespaced tool name: ``mcp__<server>__<tool>``."""
    return f"{MCP_TOOL_PREFIX}{server}__{tool}"


def parse_mcp_tool_name(namespaced: str) -> tuple[str, str] | None:
    """Parse ``mcp__<server>__<tool>`` → (server, tool) or *None*."""
    if not namespaced.startswith(MCP_TOOL_PREFIX):
        return None
    rest = namespaced[len(MCP_TOOL_PREFIX):]
    parts = rest.split("__", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def is_mcp_tool(name: str) -> bool:
    """Return True if *name* looks like a namespaced MCP tool."""
    return name.startswith(MCP_TOOL_PREFIX)


# ---------------------------------------------------------------------------
# MCP Tool → OpenAI function-tool conversion
# ---------------------------------------------------------------------------

def _mcp_tool_to_openai(server_name: str, tool: Any) -> dict[str, Any]:
    """Convert an MCP ``Tool`` object to an OpenAI function-tool dict.

    The tool name is namespaced as ``mcp__<server>__<original_name>`` so
    it can be routed back to the correct server at call time.
    """
    namespaced = mcp_tool_name(server_name, tool.name)
    parameters = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": namespaced,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }


def _call_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert an MCP ``CallToolResult`` to a plain dict for JSON serialisation."""
    # Prefer structured content if available
    if getattr(result, "structuredContent", None):
        return result.structuredContent

    texts: list[str] = []
    for block in (result.content or []):
        if hasattr(block, "text"):
            texts.append(block.text)
        elif hasattr(block, "data"):
            texts.append(f"[binary: {getattr(block, 'mimeType', 'unknown')}]")
        else:
            texts.append(str(block))

    payload: dict[str, Any] = {"result": "\n".join(texts) if texts else ""}
    if result.isError:
        payload["error"] = True
    return payload


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MCPToolRegistry:
    """Manages MCP server connections and proxies tool calls.

    Connections run on a background asyncio event loop in a daemon thread.
    The synchronous ``connect`` / ``call_tool`` / ``disconnect_all`` methods
    block until the underlying async operations complete, making this usable
    from the synchronous orchestrator code.
    """

    def __init__(self, server_configs: dict[str, MCPServerConfig] | None = None):
        self._configs: dict[str, MCPServerConfig] = server_configs or {}
        # server_name → list of OpenAI-format tool definitions
        self._tool_cache: dict[str, list[dict[str, Any]]] = {}
        # server_name → mcp.ClientSession (kept alive by _exit_stacks)
        self._sessions: dict[str, Any] = {}
        # server_name → AsyncExitStack holding the SSE + session contexts open
        self._exit_stacks: dict[str, AsyncExitStack] = {}
        # Background event loop + thread (lazy-started)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Background event-loop plumbing
    # ------------------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start a background event loop thread if not already running."""
        if self._loop is not None and self._loop.is_running():
            return self._loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, daemon=True, name="mcp-event-loop",
        )
        thread.start()
        self._loop = loop
        self._thread = thread
        return loop

    def _run_sync(self, coro: Any, *, timeout: float = 30) -> Any:
        """Schedule *coro* on the background loop and block for the result."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, server_name: str) -> None:
        """Connect to a named MCP server (lazy, idempotent).

        On success the server's tools are cached and available via
        ``list_tool_definitions``.  On failure a warning is logged and
        an empty tool list is cached so the rest of the pipeline
        continues gracefully.
        """
        if server_name in self._tool_cache:
            return  # already connected / attempted

        cfg = self._configs.get(server_name)
        if cfg is None:
            logger.warning("MCP server '%s' not found in config — skipping", server_name)
            return

        if cfg.transport not in ("sse", "streamable_http", "auto"):
            logger.warning(
                "MCP server '%s': transport '%s' not supported "
                "(use 'streamable_http', 'sse', or 'auto')",
                server_name, cfg.transport,
            )
            self._tool_cache[server_name] = []
            return

        if not cfg.url:
            logger.warning("MCP server '%s': no URL configured — skipping", server_name)
            self._tool_cache[server_name] = []
            return

        try:
            self._run_sync(self._connect_server(cfg), timeout=30)
            tool_count = len(self._tool_cache.get(server_name, []))
            logger.info(
                "MCP server '%s' connected — %d tool(s) available",
                server_name, tool_count,
            )
        except Exception as exc:
            logger.warning(
                "Failed to connect to MCP server '%s': %s — running without it",
                server_name, exc,
            )
            self._tool_cache[server_name] = []

    async def _connect_server(self, cfg: MCPServerConfig) -> None:
        """Open a connection using the configured transport, init session, cache tools."""
        transport = cfg.transport

        # "auto": try streamable_http first, fall back to SSE
        if transport in ("streamable_http", "auto"):
            try:
                await self._connect_streamable_http(cfg)
                return
            except Exception:
                if transport != "auto":
                    raise
                logger.debug(
                    "Streamable HTTP failed for '%s', falling back to SSE",
                    cfg.name,
                )

        # SSE fallback (or explicit "sse")
        await self._connect_sse(cfg)

    async def _connect_streamable_http(self, cfg: MCPServerConfig) -> None:
        """Connect via the Streamable HTTP transport."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        stack = AsyncExitStack()
        try:
            read_stream, write_stream, _get_sid = await stack.enter_async_context(
                streamablehttp_client(cfg.url, timeout=10, sse_read_timeout=300),
            )
            session: ClientSession = await stack.enter_async_context(
                ClientSession(read_stream, write_stream),
            )
            await session.initialize()

            tools_result = await session.list_tools()
            defs = [_mcp_tool_to_openai(cfg.name, t) for t in tools_result.tools]

            self._sessions[cfg.name] = session
            self._exit_stacks[cfg.name] = stack
            self._tool_cache[cfg.name] = defs
        except BaseException:
            await stack.aclose()
            raise

    async def _connect_sse(self, cfg: MCPServerConfig) -> None:
        """Open an SSE connection, initialise the MCP session, cache tools."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(
                sse_client(cfg.url, timeout=10, sse_read_timeout=300),
            )
            session: ClientSession = await stack.enter_async_context(
                ClientSession(read_stream, write_stream),
            )
            await session.initialize()

            # Fetch available tools and convert to OpenAI schema
            tools_result = await session.list_tools()
            defs = [_mcp_tool_to_openai(cfg.name, t) for t in tools_result.tools]

            # Store everything — stack keeps the connection alive
            self._sessions[cfg.name] = session
            self._exit_stacks[cfg.name] = stack
            self._tool_cache[cfg.name] = defs
        except BaseException:
            await stack.aclose()
            raise

    def disconnect(self, server_name: str) -> None:
        """Disconnect a single MCP server."""
        stack = self._exit_stacks.pop(server_name, None)
        self._sessions.pop(server_name, None)
        self._tool_cache.pop(server_name, None)
        if stack and self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    stack.aclose(), self._loop,
                )
                future.result(timeout=10)
            except Exception as exc:
                logger.debug("Error disconnecting MCP '%s': %s", server_name, exc)

    def disconnect_all(self) -> None:
        """Close all MCP sessions and stop the background event loop."""
        for name in list(self._exit_stacks):
            self.disconnect(name)

        # Shut down the background loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None

    # ------------------------------------------------------------------
    # Tool schema
    # ------------------------------------------------------------------

    def list_tool_definitions(self, server_name: str) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool definitions for one MCP server.

        Each tool's ``function.name`` is namespaced as
        ``mcp__<server>__<original_name>``.  Returns an empty list if the
        server is not connected or has no tools.
        """
        return list(self._tool_cache.get(server_name, []))

    def get_all_definitions(self, server_names: list[str]) -> list[dict[str, Any]]:
        """Return combined tool definitions for multiple MCP servers.

        Connects lazily to any server not yet connected.
        """
        defs: list[dict[str, Any]] = []
        for name in server_names:
            if name not in self._tool_cache:
                self.connect(name)
            defs.extend(self.list_tool_definitions(name))
        return defs

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on a connected MCP server (blocking).

        Returns a plain dict — either the tool result or an ``{"error": ...}``
        payload.
        """
        session = self._sessions.get(server_name)
        if session is None:
            return {"error": f"MCP server '{server_name}' is not connected"}

        try:
            result = self._run_sync(
                session.call_tool(tool_name, arguments), timeout=120,
            )
            return _call_result_to_dict(result)
        except Exception as exc:
            return {"error": f"MCP call to {server_name}/{tool_name} failed: {exc}"}

    def route_tool_call(
        self, namespaced_name: str, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Route a namespaced MCP tool call (``mcp__server__tool``).
        """
        parsed = parse_mcp_tool_name(namespaced_name)
        if parsed is None:
            return {"error": f"invalid MCP tool name: {namespaced_name}"}
        server_name, tool_name = parsed
        return self.call_tool(server_name, tool_name, arguments)
