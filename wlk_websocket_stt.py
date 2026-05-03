"""WebSocket STT plugin that streams audio to WhisperLiveKit.

Streams all audio frames as raw PCM to a WLK WebSocket endpoint
(ws://host:port/asr) and emits FINAL_TRANSCRIPT events when the server
sends {"type": "final_transcript", "text": "..."} messages.

WLK handles all VAD, segmentation, and transcription internally.
"""

from __future__ import annotations

import asyncio
import json
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions

class STT(stt.STT):
    def __init__(self, *, ws_url: str = "ws://localhost:8000/asr", sample_rate: int = 16000):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=False),
        )
        self._ws_url = ws_url
        self._sample_rate = sample_rate

    @property
    def model(self):
        return "wlk-qwen3"

    @property
    def provider(self):
        return "wlk-websocket"

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        raise NotImplementedError("use stream()")

    def stream(self, *, language=NOT_GIVEN, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return WLKSpeechStream(stt=self, conn_options=conn_options, ws_url=self._ws_url, sample_rate=self._sample_rate)


class WLKSpeechStream(stt.RecognizeStream):
    def __init__(self, *, stt, conn_options, ws_url, sample_rate):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._ws_url = ws_url
        self._sample_rate = sample_rate

    def _emit_transcript(self, text: str) -> None:
        self._event_ch.send_nowait(stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language="auto")],
        ))

    @utils.log_exceptions()
    async def _run(self) -> None:
        import websockets

        closing_ws = False

        async with websockets.connect(self._ws_url) as ws:
            config_msg = await ws.recv()
            json.loads(config_msg)

            async def send_task():
                bstream = utils.audio.AudioByteStream(
                    sample_rate=self._sample_rate, num_channels=1,
                    samples_per_channel=512,
                )

                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        frames = bstream.flush()
                    elif isinstance(data, rtc.AudioFrame):
                        frames = bstream.write(data.data.tobytes())
                    else:
                        continue

                    for frame in frames:
                        await ws.send(frame.data.tobytes())

                try:
                    await ws.send(b"")
                except Exception:
                    pass

            async def recv_task():
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        if closing_ws:
                            break
                        continue
                    except Exception:
                        if closing_ws:
                            break
                        break

                    try:
                        data = json.loads(msg)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    if data.get("type") == "ready_to_stop":
                        break

                    if data.get("type") == "final_transcript":
                        text = data.get("text", "").strip()
                        if text:
                            self._emit_transcript(text)

            tasks = [
                asyncio.create_task(send_task()),
                asyncio.create_task(recv_task()),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                closing_ws = True
                await utils.aio.gracefully_cancel(*tasks)
