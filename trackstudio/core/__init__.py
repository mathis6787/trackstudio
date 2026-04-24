"""TrackStudio core module."""

__all__ = [
    "fastapi_app",
    "ServerConfig",
    "stream_combiner_manager",
    "StreamCombinerTrack",
    "VisionAPI",
    "get_vision_api",
    "create_vision_api",
    "VisionWebSocketManager",
]


def __getattr__(name: str):
    if name == "fastapi_app":
        from .app import app as fastapi_app  # noqa: PLC0415

        return fastapi_app
    if name == "ServerConfig":
        from .config import ServerConfig  # noqa: PLC0415

        return ServerConfig
    if name in {"stream_combiner_manager", "StreamCombinerTrack"}:
        from .stream_combiner import StreamCombinerTrack, stream_combiner_manager  # noqa: PLC0415

        return {"stream_combiner_manager": stream_combiner_manager, "StreamCombinerTrack": StreamCombinerTrack}[name]
    if name in {"VisionAPI", "create_vision_api", "get_vision_api"}:
        from .vision_api import VisionAPI, create_vision_api, get_vision_api  # noqa: PLC0415

        return {"VisionAPI": VisionAPI, "create_vision_api": create_vision_api, "get_vision_api": get_vision_api}[name]
    if name == "VisionWebSocketManager":
        from .vision_websocket import VisionWebSocketManager  # noqa: PLC0415

        return VisionWebSocketManager
    raise AttributeError(name)
