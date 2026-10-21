"""JARVIS Lite — Audio Devices Enumerator and Selector.

Lists and selects input microphones and output speakers via sounddevice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import sounddevice as sd

from core.logging import get_logger

logger = get_logger("voice.devices")


@dataclass
class AudioDeviceInfo:
    index: int
    name: str
    hostapi: int
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False
    is_default_output: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioDeviceManager:
    """Manages audio I/O hardware enumeration and selection."""

    def list_input_devices(self) -> list[AudioDeviceInfo]:
        """List all available microphone input devices."""
        devices: list[AudioDeviceInfo] = []
        try:
            dev_list = sd.query_devices()
            default_in, _ = sd.default.device
            for idx, dev in enumerate(dev_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append(
                        AudioDeviceInfo(
                            index=idx,
                            name=str(dev.get("name", f"Input Device {idx}")),
                            hostapi=int(dev.get("hostapi", 0)),
                            max_input_channels=int(dev.get("max_input_channels", 0)),
                            max_output_channels=int(dev.get("max_output_channels", 0)),
                            default_samplerate=float(dev.get("default_samplerate", 16000.0)),
                            is_default_input=(idx == default_in),
                        )
                    )
        except Exception as exc:
            logger.warn("list_input_devices_failed", error=str(exc))
        return devices

    def list_output_devices(self) -> list[AudioDeviceInfo]:
        """List all available speaker/headphone output devices."""
        devices: list[AudioDeviceInfo] = []
        try:
            dev_list = sd.query_devices()
            _, default_out = sd.default.device
            for idx, dev in enumerate(dev_list):
                if dev.get("max_output_channels", 0) > 0:
                    devices.append(
                        AudioDeviceInfo(
                            index=idx,
                            name=str(dev.get("name", f"Output Device {idx}")),
                            hostapi=int(dev.get("hostapi", 0)),
                            max_input_channels=int(dev.get("max_input_channels", 0)),
                            max_output_channels=int(dev.get("max_output_channels", 0)),
                            default_samplerate=float(dev.get("default_samplerate", 44100.0)),
                            is_default_output=(idx == default_out),
                        )
                    )
        except Exception as exc:
            logger.warn("list_output_devices_failed", error=str(exc))
        return devices

    def set_default_devices(self, input_idx: int | None = None, output_idx: int | None = None) -> None:
        """Set system active input/output audio devices."""
        try:
            cur_in, cur_out = sd.default.device
            new_in = input_idx if input_idx is not None else cur_in
            new_out = output_idx if output_idx is not None else cur_out
            sd.default.device = (new_in, new_out)
            logger.info("audio_devices_configured", input_device=new_in, output_device=new_out)
        except Exception as exc:
            logger.warn("failed_to_set_audio_device", error=str(exc))


audio_devices = AudioDeviceManager()
