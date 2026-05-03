"""Segment-and-Transcribe streaming strategy."""

import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

MAX_SEGMENT_DURATION_S = 30.0
MIN_SEGMENT_DURATION_S = 0.3
DUMP_SEGMENTS_DIR = "/tmp/wlk_segments"

# Context mode: "none", "keywords", "last_segment", "full_history"
CONTEXT_MODE = "none"
KEYWORDS_PROMPT = "restaurant reservation, party size, ten people, English"


class SegmentTranscribeProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0.0
        self.committed: List[ASRToken] = []
        self.end = 0.0
        self.last_segment_text = ""
        self._segment_counter = 0
        if DUMP_SEGMENTS_DIR:
            os.makedirs(DUMP_SEGMENTS_DIR, exist_ok=True)
            for f in os.listdir(DUMP_SEGMENTS_DIR):
                os.remove(os.path.join(DUMP_SEGMENTS_DIR, f))
            logger.info("Segment audio dump enabled: %s", DUMP_SEGMENTS_DIR)

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: Optional[float] = None):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        if audio_stream_end_time is not None:
            self.end = audio_stream_end_time

    def process_iter(self) -> Tuple[List[ASRToken], float]:
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        if buffer_duration >= MAX_SEGMENT_DURATION_S:
            return self._transcribe_and_reset()
        return [], self.end

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        if self.audio_buffer.size == 0:
            return [], self.end
        return self._transcribe_and_reset()

    def end_silence(self, silence_duration=None, offset=None):
        if silence_duration and silence_duration >= 5:
            self.buffer_time_offset = (offset or 0.0) + silence_duration

    def get_buffer(self) -> Transcript:
        return Transcript()

    def finish(self) -> Tuple[List[ASRToken], float]:
        if self.audio_buffer.size == 0:
            return [], self.end
        return self._transcribe_and_reset()

    def new_speaker(self, change_speaker):
        if self.audio_buffer.size > 0:
            self._transcribe_and_reset()
        self.buffer_time_offset = change_speaker.start

    def _build_init_prompt(self) -> str:
        if CONTEXT_MODE == "none":
            return ""
        elif CONTEXT_MODE == "keywords":
            return KEYWORDS_PROMPT
        elif CONTEXT_MODE == "last_segment":
            return self.last_segment_text[:200] if self.last_segment_text else ""
        elif CONTEXT_MODE == "full_history":
            if not self.committed:
                return ""
            parts = []
            char_count = 0
            for t in reversed(self.committed):
                char_count += len(t.text) + 1
                if char_count > 200:
                    break
                parts.append(t.text)
            return self.asr.sep.join(reversed(parts)).strip()
        return ""

    def _transcribe_and_reset(self) -> Tuple[List[ASRToken], float]:
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        if buffer_duration < MIN_SEGMENT_DURATION_S:
            logger.info("Skipping short segment: %.2fs", buffer_duration)
            self.buffer_time_offset += buffer_duration
            self.audio_buffer = np.array([], dtype=np.float32)
            return [], self.end

        init_prompt = self._build_init_prompt()
        self._segment_counter += 1
        logger.info("Transcribing segment %d: %.2fs at offset %.2fs (context_mode=%s, prompt=%r)",
                    self._segment_counter, buffer_duration, self.buffer_time_offset, CONTEXT_MODE, init_prompt[:80])

        if DUMP_SEGMENTS_DIR:
            dump_path = os.path.join(DUMP_SEGMENTS_DIR, f"seg_{self._segment_counter:03d}_{buffer_duration:.2f}s.wav")
            sf.write(dump_path, self.audio_buffer, self.SAMPLING_RATE)
            logger.info("Dumped segment audio: %s", dump_path)

        import time as _time
        _t0 = _time.perf_counter()
        result = self.asr.transcribe(self.audio_buffer, init_prompt=init_prompt)
        _dur = _time.perf_counter() - _t0
        tokens = self.asr.ts_words(result)
        adjusted = [t.with_offset(self.buffer_time_offset) for t in tokens]
        segment_text = self.asr.sep.join(t.text for t in adjusted)
        print(f'[PERF] seg {self._segment_counter}: {_dur*1000:.0f}ms for {buffer_duration:.2f}s audio (RTF={_dur/buffer_duration:.3f}) text="{segment_text}"', flush=True)
        with open('/tmp/wlk_transcripts.log', 'a') as _tf:
            _tf.write(f'[SEG {self._segment_counter}] {segment_text}' + chr(10))
            _tf.flush()
        self.committed.extend(adjusted)

        segment_text = self.asr.sep.join(t.text for t in adjusted)
        self.last_segment_text = segment_text

        if adjusted:
            logger.info("Segment [%.1f-%.1fs]: %s",
                        self.buffer_time_offset,
                        self.buffer_time_offset + buffer_duration,
                        segment_text[:120])

        self.buffer_time_offset += buffer_duration
        self.audio_buffer = np.array([], dtype=np.float32)
        return adjusted, self.end
