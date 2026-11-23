# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from .._fvdb_cpp import CheckboxView as _CheckboxViewCpp
from .._fvdb_cpp import NumberView as _NumberViewCpp
from .._fvdb_cpp import SliderView as _SliderViewCpp
from .._fvdb_cpp import TextView as _TextViewCpp
from ._viewer_server import _get_viewer_server_cpp

_T = TypeVar("_T")


class _WidgetView(Generic[_T]):
    """Common base for the four widget handles. Tracks ``on_update`` callbacks
    plus the value last seen during a successful poll so we know when to fire
    them."""

    __PRIVATE__ = object()

    def __init__(self, scene_name: str, name: str):
        self._scene_name = scene_name
        self._name = name
        self._callbacks: list[Callable[[_T], None]] = []
        # Sentinel until the first successful poll, after which we cache the
        # most recent value seen so subsequent polls can detect changes.
        self._last_value: Any = _WidgetView._UNSET

    _UNSET = object()

    @property
    def name(self) -> str:
        return self._name

    @property
    def scene_name(self) -> str:
        return self._scene_name

    @property
    def value(self) -> _T:
        raise NotImplementedError

    @value.setter
    def value(self, new_value: _T) -> None:
        raise NotImplementedError

    def on_update(self, callback: Callable[[_T], None]) -> Callable[[_T], None]:
        """
        Register a callback that fires when this widget's value changes.

        Can be used as a decorator::

            blend = scene.add_slider("blend", 0.0, 1.0, 0.5)

            @blend.on_update
            def _(new_value: float) -> None:
                print("blend changed to", new_value)

        Args:
            callback: A callable taking the new widget value.

        Returns:
            The same ``callback``, so this method can be used as a decorator.
        """
        self._callbacks.append(callback)
        return callback

    def remove_on_update(self, callback: Callable[[_T], None]) -> None:
        """Remove a previously-registered ``on_update`` callback."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def _fire_on_update_if_changed(self) -> bool:
        """
        Read the current value and, if it differs from the last polled value,
        fire all registered ``on_update`` callbacks. Returns True if a change
        was detected (and callbacks were fired).
        """
        current = self.value
        previous = self._last_value
        self._last_value = current
        if previous is _WidgetView._UNSET:
            # First poll: prime the cache without firing callbacks.
            return False
        if previous == current:
            return False
        for cb in list(self._callbacks):
            cb(current)
        return True

    def _fire_on_submit_if_changed(self) -> bool:
        """Override hook for widgets with a separate submit event."""
        return False


class SliderView(_WidgetView[float]):
    """Handle to a float slider widget in the editor's `Scene Params` window."""

    def __init__(
        self,
        scene_name: str,
        name: str,
        min: float = 0.0,
        max: float = 1.0,
        initial: float = 0.0,
        step: float = 0.01,
        _private: Any = None,
    ):
        if _private is not _WidgetView.__PRIVATE__:
            raise ValueError("SliderView constructor is private. Use Scene.add_slider() instead.")
        super().__init__(scene_name=scene_name, name=name)
        server = _get_viewer_server_cpp()
        server.add_slider(scene_name, name, float(min), float(max), float(initial), float(step))

    def _get_view(self) -> _SliderViewCpp:
        server = _get_viewer_server_cpp()
        return server.get_slider(self._name)

    @property
    def value(self) -> float:
        return float(self._get_view().value)

    @value.setter
    def value(self, new_value: float) -> None:
        view = self._get_view()
        if not (view.min <= new_value <= view.max):
            raise ValueError(f"Slider {self._name!r} value {new_value} is outside [{view.min}, {view.max}]")
        view.value = float(new_value)

    @property
    def min(self) -> float:
        """Minimum slider value."""
        return float(self._get_view().min)

    @property
    def max(self) -> float:
        """Maximum slider value."""
        return float(self._get_view().max)

    @property
    def step(self) -> float:
        """Slider step size."""
        return float(self._get_view().step)


class NumberView(_WidgetView[float]):
    """Handle to a float numeric drag widget in the editor's `Scene Params` window."""

    def __init__(
        self,
        scene_name: str,
        name: str,
        initial: float = 0.0,
        min: float | None = None,
        max: float | None = None,
        step: float = 0.01,
        _private: Any = None,
    ):
        if _private is not _WidgetView.__PRIVATE__:
            raise ValueError("NumberView constructor is private. Use Scene.add_number() instead.")
        super().__init__(scene_name=scene_name, name=name)
        server = _get_viewer_server_cpp()
        server.add_number(
            scene_name,
            name,
            float(initial),
            min is not None,
            float(0.0 if min is None else min),
            max is not None,
            float(0.0 if max is None else max),
            float(step),
        )

    def _get_view(self) -> _NumberViewCpp:
        server = _get_viewer_server_cpp()
        return server.get_number(self._name)

    @property
    def value(self) -> float:
        return float(self._get_view().value)

    @value.setter
    def value(self, new_value: float) -> None:
        view = self._get_view()
        if view.has_min and new_value < view.min:
            raise ValueError(f"Number {self._name!r} value {new_value} is below min {view.min}")
        if view.has_max and new_value > view.max:
            raise ValueError(f"Number {self._name!r} value {new_value} is above max {view.max}")
        view.value = float(new_value)

    @property
    def min(self) -> float | None:
        """Minimum value (or None when unbounded)."""
        view = self._get_view()
        return float(view.min) if view.has_min else None

    @property
    def max(self) -> float | None:
        """Maximum value (or None when unbounded)."""
        view = self._get_view()
        return float(view.max) if view.has_max else None

    @property
    def step(self) -> float:
        """Drag widget step size."""
        return float(self._get_view().step)


class TextView(_WidgetView[str]):
    """Handle to a text input widget in the editor's `Scene Params` window.

    With ``commit_on_enter=True`` (via :meth:`Scene.add_text`), an
    Enter-driven :meth:`on_submit` callback becomes available alongside
    the per-keystroke :meth:`on_update`.
    """

    def __init__(
        self,
        scene_name: str,
        name: str,
        initial: str = "",
        max_length: int = 256,
        commit_on_enter: bool = False,
        _private: Any = None,
    ):
        if _private is not _WidgetView.__PRIVATE__:
            raise ValueError("TextView constructor is private. Use Scene.add_text() instead.")
        super().__init__(scene_name=scene_name, name=name)
        self._commit_on_enter = bool(commit_on_enter)
        self._submit_callbacks: list[Callable[[str], None]] = []
        self._last_submit_counter: Any = _WidgetView._UNSET
        server = _get_viewer_server_cpp()
        server.add_text(scene_name, name, initial, int(max_length), self._commit_on_enter)

    def _get_view(self) -> _TextViewCpp:
        server = _get_viewer_server_cpp()
        return server.get_text(self._name)

    @property
    def value(self) -> str:
        return str(self._get_view().value)

    @value.setter
    def value(self, new_value: str) -> None:
        if not isinstance(new_value, str):
            raise TypeError(f"Text widget {self._name!r} expects a str value, got {type(new_value).__name__}")
        view = self._get_view()
        encoded = new_value.encode("utf-8")
        if len(encoded) > view.max_length - 1:
            raise ValueError(
                f"Text widget {self._name!r} value is too long: {len(encoded)} bytes, "
                f"max {view.max_length - 1} (excluding NUL terminator)"
            )
        view.value = new_value

    @property
    def max_length(self) -> int:
        """Capacity of the underlying ``char[N]`` buffer in bytes."""
        return int(self._get_view().max_length)

    @property
    def commit_on_enter(self) -> bool:
        """True iff this widget was created with ``commit_on_enter=True``."""
        return self._commit_on_enter

    def on_submit(self, callback: Callable[[str], None]) -> Callable[[str], None]:
        """
        Register a callback that fires when the user presses Enter on this
        text input. Requires ``commit_on_enter=True``; otherwise raises
        :class:`RuntimeError`. Usable as a decorator::

            @query.on_submit
            def _(value: str) -> None:
                print("submitted:", value)
        """
        if not self._commit_on_enter:
            raise RuntimeError(f"on_submit requires commit_on_enter=True (widget {self._name!r} was not)")
        self._submit_callbacks.append(callback)
        return callback

    def remove_on_submit(self, callback: Callable[[str], None]) -> None:
        """Remove a previously-registered ``on_submit`` callback."""
        try:
            self._submit_callbacks.remove(callback)
        except ValueError:
            pass

    def _fire_on_submit_if_changed(self) -> bool:
        if not self._commit_on_enter:
            return False
        current = int(self._get_view().submit_counter)
        previous = self._last_submit_counter
        self._last_submit_counter = current
        if previous is _WidgetView._UNSET or previous == current:
            return False
        value = self.value
        for cb in list(self._submit_callbacks):
            cb(value)
        return True


class CheckboxView(_WidgetView[bool]):
    """Handle to a checkbox widget in the editor's `Scene Params` window."""

    def __init__(
        self,
        scene_name: str,
        name: str,
        initial: bool = False,
        _private: Any = None,
    ):
        if _private is not _WidgetView.__PRIVATE__:
            raise ValueError("CheckboxView constructor is private. Use Scene.add_checkbox() instead.")
        super().__init__(scene_name=scene_name, name=name)
        server = _get_viewer_server_cpp()
        server.add_checkbox(scene_name, name, bool(initial))

    def _get_view(self) -> _CheckboxViewCpp:
        server = _get_viewer_server_cpp()
        return server.get_checkbox(self._name)

    @property
    def value(self) -> bool:
        return bool(self._get_view().value)

    @value.setter
    def value(self, new_value: bool) -> None:
        self._get_view().value = bool(new_value)
