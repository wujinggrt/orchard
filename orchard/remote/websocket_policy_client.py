import time
import asyncio
import websockets
import websockets.sync.client as _client
import websockets.asyncio.client as _async_client
from orchard.utils.logging_utils import get_logger
from orchard.remote import msgpack_numpy

logger = get_logger(__name__)


class WebsocketPolicyClient:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[_client.ClientConnection, dict]:
        logger.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = _client.connect(
                    uri=self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    def request(self, *, obs: dict) -> dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)


class AsyncWebsocketPolicyClient:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws: _async_client.ClientConnection | None = None
        self._server_metadata: dict | None = None

    async def connect(self) -> None:
        """Connect to the server asynchronously."""
        self._ws, self._server_metadata = await self._wait_for_server()

    async def get_server_metadata(self) -> dict:
        """Get server metadata after connection."""
        if self._server_metadata is None:
            raise RuntimeError("Not connected to server yet. Call connect() first.")
        return self._server_metadata

    async def _wait_for_server(self) -> tuple[_async_client.ClientConnection, dict]:
        """Wait for server to become available and establish connection."""
        logger.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = await _async_client.connect(
                    uri=self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(await conn.recv())
                return conn, metadata
            except (
                ConnectionRefusedError,
                OSError,
                websockets.exceptions.WebSocketException,
            ) as e:
                logger.info(f"Still waiting for server... Error: {e}")
                await asyncio.sleep(5)

    async def request(self, *, obs: dict) -> dict:  # noqa: UP006
        """Send observation to server and get action asynchronously."""
        if self._ws is None:
            raise RuntimeError("Not connected to server. Call connect() first.")

        data = self._packer.pack(obs)
        await self._ws.send(data)

        response = await self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
            self._server_metadata = None
