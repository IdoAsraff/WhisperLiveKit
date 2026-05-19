"""
WebSocket STT plugin that streams audio.
"""

from __future__ import annotations

import asyncio
import json

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions


class STT(stt.STT):
    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8000/v1/realtime",
        sample_rate: int = 16000,
        api_key: str = "",
        model: str = "",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=False),
        )
        self._ws_url = ws_url
        self._sample_rate = sample_rate
        self._api_key = api_key
        self._model_name = model

    @property
    def model(self):
        return self._model_name

    @property
    def provider(self):
        return "Crusoe AI"

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        raise NotImplementedError("use stream()")

    def stream(self, *, language=NOT_GIVEN, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return CrusoeSpeechStream(
            stt=self, conn_options=conn_options, ws_url=self._ws_url,
            sample_rate=self._sample_rate, api_key=self._api_key,
            model=self._model_name,
        )


class CrusoeSpeechStream(stt.RecognizeStream):
    def __init__(self, *, stt, conn_options, ws_url, sample_rate, api_key, model):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._ws_url = ws_url
        self._sample_rate = sample_rate
        self._api_key = api_key
        self._model = model

    def _emit_transcript(self, text: str, language: str = "en") -> None:
        self._event_ch.send_nowait(stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language)],
        ))

    async def _get_ephemeral_token(self) -> str:
        """Fetch a single-use ephemeral token from the gateway."""
        import aiohttp

        base = self._ws_url.replace("ws://", "http://").replace("wss://", "https://")
        # Strip path to get base URL, then append auth endpoint
        from urllib.parse import urlparse
        parsed = urlparse(base)
        auth_url = f"{parsed.scheme}://{parsed.netloc}/v1/realtime/auth_token"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                auth_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                ssl=False,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["token"]

    async def _build_ws_url(self) -> str:
        """Build the final WebSocket URL, with ephemeral token if auth is configured."""
        from urllib.parse import urlencode, urlparse, urlunparse

        parsed = urlparse(self._ws_url)
        params = {}

        if self._api_key:
            token = await self._get_ephemeral_token()
            params["token"] = token

        if self._model:
            params["model"] = self._model

        if params:
            separator = "&" if parsed.query else ""
            new_query = parsed.query + separator + urlencode(params)
            parsed = parsed._replace(query=new_query)

        return urlunparse(parsed)

    @utils.log_exceptions()
    async def _run(self) -> None:
        import websockets

        closing_ws = False
        ws_url = await self._build_ws_url()

        ws_kwargs = {}
        if ws_url.startswith("wss://"):
            import ssl as _ssl
            ctx = _ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            ws_kwargs["ssl"] = ctx

        async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None, **ws_kwargs) as ws:
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
                            lang = data.get("language", "en")
                            self._emit_transcript(text, lang)

            tasks = [
                asyncio.create_task(send_task()),
                asyncio.create_task(recv_task()),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                closing_ws = True
                await utils.aio.gracefully_cancel(*tasks)
