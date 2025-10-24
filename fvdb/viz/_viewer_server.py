# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import warnings
import webbrowser

from .._Cpp import Viewer as ViewerCpp

# Global viewer server. Create by calling init()
_viewer_server_cpp: ViewerCpp | None = None


def _get_viewer_server_cpp() -> ViewerCpp:
    """
    Get the global viewer server C++ instance or raise a :class:`RuntimeError` if it is not initialized.

    Returns:
        viewer_server (ViewerCpp): The global viewer server C++ instance.

    """
    global _viewer_server_cpp
    if _viewer_server_cpp is None:
        raise RuntimeError("Viewer server is not initialized. Call fvdb.viz.init() first.")
    return _viewer_server_cpp


def init(ip_address: str = "127.0.0.1", port: int = 8080, vk_device_id: int = 0, verbose: bool = False):
    """
    Initialize the viewer web-server on the given IP address and port. You must call this function
    first before visualizing any scenes.

    Example usage:

    .. code-block:: python

        import fvdb

        # Initialize the viewer server on localhost:8080
        fvdb.viz.init(ip_address="127.0.0.1", port=8080)

        # Add a scene to the viewer with a point cloud in the scene
        scene = fvdb.viz.Scene("My Scene")
        scene.add_point_cloud(...)

        # Show the viewer in the browser or inline in a Jupyter notebook
        fvdb.viz.show()

    .. note::

        If the viewer server is already initialized, this function will do nothing and
        will print a warning message.

    Args:
        ip_address (str): The IP address to bind the viewer server to. Default is ``"127.0.0.1"``.
        port (int): The port to bind the viewer server to. Default is ``8080``.
        vk_device_id (int): The Vulkan device ID to use for rendering. Default is ``0``.
        verbose (bool): If True, the viewer server will print verbose output to the console. Default is ``False``.
    """
    global _viewer_server_cpp
    if _viewer_server_cpp is None:
        import nanovdb_editor as editor

        compiler = editor.Compiler()
        compute = editor.Compute(compiler)
        di = compute.device_interface()
        di.create_device_manager(enable_validation=False)
        try:
            di.create_device(device_index=vk_device_id, enable_external_usage=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Vulkan device with ID {vk_device_id}. You may have an incompatible version of Vulkan installed."
            ) from e
        _viewer_server_cpp = ViewerCpp(ip_address=ip_address, port=port, device_id=vk_device_id, verbose=verbose)
    else:
        warnings.warn(
            f"Viewer server is already initialized with IP = {_viewer_server_cpp.ip_address()} and port = {_viewer_server_cpp.port()}."
        )


def show():
    """
    Show an interactive viewer in the browser or inline in a Jupyter notebook.

    Example usage:

    .. code-block:: python

        import fvdb

        # Initialize the viewer server on localhost:8080
        fvdb.viz.init(ip_address="127.0.0.1", port=8080)

        # Add a scene to the viewer with a point cloud in the scene
        scene = fvdb.viz.Scene("My Scene")
        scene.add_point_cloud(...)

        # Show the viewer in the browser or inline in a Jupyter notebook
        fvdb.viz.show()

    .. note::
        You must call :func:`fvdb.viz.init()` before calling this function. If the viewer server
        is not initialized, this function will raise a RuntimeError.
    """
    viewer_server = _get_viewer_server_cpp()
    viewer_server_ip: str = viewer_server.ip_address()
    viewer_server_port: int = viewer_server.port()
    url = f"http://{viewer_server_ip}:{viewer_server_port}"

    try:
        from IPython import get_ipython
        from IPython.display import IFrame, display

        if get_ipython() is not None:
            display(IFrame(src=url, width="100%", height="600px"))
            return
    except ImportError:
        pass

    webbrowser.open_new_tab(url)


def reset():
    """
    Reset the viewer server state. This will clear all scenes and views and ads back the default scene.
    """
    viewer_server = _get_viewer_server_cpp()
    viewer_server.reset()


def remove_scene(scene_name: str):
    """
    Remove a scene from the viewer server.

    Args:
        scene_name (str): The name of the scene to remove.
    """
    viewer_server = _get_viewer_server_cpp()
    viewer_server.remove_scene(scene_name)


def remove_view(scene_name: str, name: str):
    """
    Remove a view from the viewer server.

    Args:
        scene_name (str): The name of the scene view belongs to.
        name (str): The name of the view to remove.
    """
    viewer_server = _get_viewer_server_cpp()
    viewer_server.remove_view(scene_name, name)
