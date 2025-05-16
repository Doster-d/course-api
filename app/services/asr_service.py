import logging
import os
import traceback
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

try:
    from app.services.whisper_online import (
        FasterWhisperASR,
        OnlineASRProcessor,
    )

    WHISPER_STREAMING_AVAILABLE = True
except ImportError:
    logger.warning(
        "whisper_streaming not available. Install with: pip install git+https://github.com/ufal/whisper_streaming.git"
    )
    WHISPER_STREAMING_AVAILABLE = False

try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    logger.warning(
        "faster-whisper not available. Install with: pip install faster-whisper"
    )
    FASTER_WHISPER_AVAILABLE = False


class ASRService:
    """Service for automatic speech recognition using Whisper.

    This service provides real-time speech-to-text functionality using either
    the streaming Whisper API or direct Faster Whisper model. It supports
    incremental audio processing and both transcription and translation modes.

    Attributes:
        source_language: Language code of the input speech
        target_language: Language code for the output text
        online_processor: Streaming ASR processor when using Whisper Streaming
        asr_model: The underlying ASR model handler
        whisper_model: Direct Faster Whisper model instance for fallback
        audio_buffer: Internal buffer for audio data when using direct mode
    """

    def __init__(
        self,
        source_language: str = "ru",
        target_language: str = "ru",
        use_vad: bool = True,
        vad_chunk_size: float = 0.1,
        buffer_trimming: str = "segment",
        buffer_trimming_sec: float = 15.0,
    ):
        """Initialize the ASR service.

        Args:
            source_language: Source language code (default: "ru")
            target_language: Target language code (default: "ru")
            use_vad: Whether to use Voice Activity Detection (default: True)
            vad_chunk_size: Size of audio chunks for VAD in seconds (default: 0.1)
            buffer_trimming: Buffer trimming strategy - "segment" or "sentence" (default: "segment")
            buffer_trimming_sec: Buffer trimming length threshold in seconds (default: 15.0)
        """
        self.source_language = source_language
        self.target_language = target_language
        self.use_vad = use_vad
        self.vad_chunk_size = vad_chunk_size
        self.buffer_trimming = buffer_trimming
        self.buffer_trimming_sec = buffer_trimming_sec
        self.online_processor = None
        self.asr_model = None
        self.whisper_model = None
        self.audio_buffer = bytearray()

        self._initialize_whisper()

    def _initialize_whisper(self):
        """Initialize the Whisper ASR model and processor.

        Attempts to initialize using Whisper Streaming first, and falls back to
        direct Faster Whisper if streaming is not available. Configures VAD
        (Voice Activity Detection) if enabled.

        Raises:
            Exception: Logs error if initialization fails but doesn't raise to caller
        """
        try:
            if WHISPER_STREAMING_AVAILABLE:
                self.asr_model = FasterWhisperASR(
                    self.source_language, settings.WHISPER_MODEL
                )

                if self.source_language != self.target_language:
                    self.asr_model.set_translate_task()

                if self.use_vad:
                    self.asr_model.use_vad()

                self.online_processor = OnlineASRProcessor(
                    self.asr_model,
                    buffer_trimming=(
                        self.buffer_trimming,
                        self.buffer_trimming_sec,
                    ),
                )

                logger.info(
                    f"Initialized Whisper Streaming with model {settings.WHISPER_MODEL}"
                    f" (VAD: {self.use_vad}, buffer_trimming: {self.buffer_trimming})"
                )
            elif FASTER_WHISPER_AVAILABLE:
                self.whisper_model = WhisperModel(
                    model_size_or_path=settings.WHISPER_MODEL,
                    device="cuda"
                    if settings.CUDA_VISIBLE_DEVICES is not None
                    else "cpu",
                    compute_type="float16"
                    if settings.CUDA_VISIBLE_DEVICES is not None
                    else "float32",
                )
                logger.info(
                    f"Initialized Faster Whisper directly with model {settings.WHISPER_MODEL}"
                )
            else:
                logger.error(
                    "Neither whisper_streaming nor faster-whisper is available"
                )
        except Exception as e:
            logger.error(f"Failed to initialize Whisper ASR: {str(e)}")
            logger.error(traceback.format_exc())
            self.asr_model = None
            self.online_processor = None

    def insert_audio_chunk(self, audio_chunk: bytes) -> None:
        """Insert an audio chunk for processing.

        Adds a chunk of audio data to the processing queue. If using streaming mode,
        sends it directly to the processor. Otherwise, adds it to the internal buffer.

        Args:
            audio_chunk: Raw audio data chunk (typically 16-bit PCM)
        """
        if self.online_processor:
            try:
                self.online_processor.insert_audio_chunk(audio_chunk)
            except Exception as e:
                logger.error(f"Error inserting audio chunk: {str(e)}")
        else:
            self.audio_buffer.extend(audio_chunk)

    def process_iter(self) -> Optional[str]:
        """Process the current audio buffer and return transcription.

        In streaming mode, processes the latest audio chunks incrementally.
        In direct mode, processes the entire buffer when it's large enough.

        Returns:
            Transcribed text or None if processing fails or no text is available
        """
        if self.online_processor:
            try:
                result = self.online_processor.process_iter()
                return result if result else None
            except Exception as e:
                logger.error(f"Error in ASR processing: {str(e)}")
                logger.error(traceback.format_exc())
                return None
        elif (
            self.whisper_model and len(self.audio_buffer) > 16000
        ):  # Minimum 1 second at 16kHz
            try:
                audio_data = (
                    np.frombuffer(self.audio_buffer, dtype=np.int16).astype(
                        np.float32
                    )
                    / 32768.0
                )

                segments, _ = self.whisper_model.transcribe(
                    audio_data,
                    language=self.source_language,
                    task="transcribe"
                    if self.source_language == self.target_language
                    else "translate",
                )

                text = " ".join([segment.text for segment in segments])

                self.audio_buffer = bytearray()

                return text
            except Exception as e:
                logger.error(
                    f"Error in direct Faster Whisper processing: {str(e)}"
                )
                return None
        return None

    def finish(self) -> Optional[str]:
        """Finish processing and return final transcription.

        Completes processing of all buffered audio and returns the final transcription.
        This should be called when the audio stream is complete.

        Returns:
            Final transcribed text or None if processing fails
        """
        if self.online_processor:
            try:
                return self.online_processor.finish()
            except Exception as e:
                logger.error(f"Error in ASR finish: {str(e)}")
                logger.error(traceback.format_exc())
                return None
        elif self.whisper_model and len(self.audio_buffer) > 0:
            try:
                audio_data = (
                    np.frombuffer(self.audio_buffer, dtype=np.int16).astype(
                        np.float32
                    )
                    / 32768.0
                )

                segments, _ = self.whisper_model.transcribe(
                    audio_data,
                    language=self.source_language,
                    task="transcribe"
                    if self.source_language == self.target_language
                    else "translate",
                )

                text = " ".join([segment.text for segment in segments])

                self.audio_buffer = bytearray()

                return text
            except Exception as e:
                logger.error(
                    f"Error in direct Faster Whisper finish: {str(e)}"
                )
                return None
        return None

    def cleanup(self) -> None:
        """Clean up resources.

        Resets the processor and clears audio buffers. Should be called when
        the service is no longer needed or before processing a new session.
        """
        if self.online_processor:
            try:
                self.online_processor.init()  # Reset the processor
            except Exception as e:
                logger.error(f"Error cleaning up ASR processor: {str(e)}")

        self.audio_buffer = bytearray()
