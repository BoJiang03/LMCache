# SPDX-License-Identifier: Apache-2.0
"""Protocol envelope for out-of-tree server-module extensions."""

# Third Party
import msgspec


def unknown_method_error(method: str) -> str:
    """Return the error a server answers when no module serves ``method``.

    Args:
        method: The namespaced server-module method.

    Returns:
        The ``ServerModuleCallResponse.error`` text.
    """
    return "unknown server module method: %s" % method


class ServerModuleCallRequest(msgspec.Struct):
    """Opaque request envelope dispatched by namespaced extension method.

    Args:
        method: Namespaced extension method, such as
            ``"my_package.echo"``.
        payload: Plugin-owned serialized payload bytes.
    """

    method: str
    payload: bytes


class ServerModuleCallResponse(msgspec.Struct):
    """Opaque response envelope for extension method calls.

    Args:
        success: Whether the extension handler completed successfully.
        payload: Plugin-owned serialized response bytes.
        error: Human-readable error string when ``success`` is false.
    """

    success: bool
    payload: bytes
    error: str
