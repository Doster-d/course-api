import logging
import os
import traceback
from typing import Optional

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


class ASRService:
    """Service for automatic speech recognition using Whisper.

    This service provides real-time speech-to-text functionality using
    the streaming Whisper API. It supports incremental audio processing 
    and both transcription and translation modes.

    Attributes:
        source_language: Language code of the input speech
        target_language: Language code for the output text
        online_processor: Streaming ASR processor
        asr_model: The underlying ASR model handler
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

        self._initialize_whisper()

    def _initialize_whisper(self):
        """Initialize the Whisper ASR model and processor.

        Configures the streaming ASR with VAD (Voice Activity Detection) if enabled.

        Raises:
            Exception: Logs error if initialization fails but doesn't raise to caller
        """
        if not WHISPER_STREAMING_AVAILABLE:
            logger.error("whisper_streaming is not available")
            return

        try:
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
        except Exception as e:
            logger.error(f"Failed to initialize Whisper ASR: {str(e)}")
            logger.error(traceback.format_exc())
            self.asr_model = None
            self.online_processor = None

    def insert_audio_chunk(self, audio_chunk: bytes) -> None:
        """Insert an audio chunk for processing.

        Adds a chunk of audio data to the processing queue and sends it directly to the processor.

        Args:
            audio_chunk: Raw audio data chunk (typically 16-bit PCM)
        """
        try:
            self.online_processor.insert_audio_chunk(audio_chunk)
        except Exception as e:
            logger.error(f"Error inserting audio chunk: {str(e)}")

    def process_iter(self) -> Optional[str]:
        """Process the current audio buffer and return transcription.

        Processes the latest audio chunks incrementally.

        Returns:
            Transcribed text or None if processing fails or no text is available
        """
        try:
            result = self.online_processor.process_iter()
            return result if result else None
        except Exception as e:
            logger.error(f"Error in ASR processing: {str(e)}")
            logger.error(traceback.format_exc())
        return None

    def finish(self) -> Optional[str]:
        """Finish processing and return final transcription.

        Completes processing of all buffered audio and returns the final transcription.
        This should be called when the audio stream is complete.

        Returns:
            Final transcribed text or None if processing fails
        """
        try:
            return self.online_processor.finish()
        except Exception as e:
            logger.error(f"Error in ASR finish: {str(e)}")
            logger.error(traceback.format_exc())
        return None

    def cleanup(self) -> None:
        """Clean up resources.

        Resets the processor. Should be called when
        the service is no longer needed or before processing a new session.
        """
        try:
            self.online_processor.init()  # Reset the processor
        except Exception as e:
            logger.error(f"Error cleaning up ASR processor: {str(e)}")
