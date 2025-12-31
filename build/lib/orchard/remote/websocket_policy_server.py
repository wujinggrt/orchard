import asyncio
import http
import time
import traceback
import msgpack_numpy
from typing import Callable
import websockets.asyncio.server as _server
import websockets.frames
from orchard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        *,
        policy: Callable[[dict], dict],
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Args:
            policy: A function that takes a dict of observations and returns a dict of predictions.
            host: The host to serve on. 注意，localhost 只能访问本机，用 0.0.0.0 才能监听所有网卡。
            port: The port to serve on. If None, a random port will be chosen.
            metadata: A dict of metadata to send to the client.
        """
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    # 暴露给外界
    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            handler=self._handler,
            host=self._host,
            port=self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(f"Serving on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        # 每个 handler 使用一个 Packer，保障并发安全
        packer = msgpack_numpy.Packer()

        await websocket.send(message=packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                predictions = self._policy(obs)
                infer_time = time.monotonic() - infer_time

                predictions["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    # 客户端两次发起请求的时间间隔。即服务端发送后，重新收到客户端的请求时间间隔
                    predictions["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(message=packer.pack(predictions))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(message=traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(status=http.HTTPStatus.OK, text="OK\n")
    # Continue with the normal request handling.
    return None
