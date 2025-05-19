#!/usr/bin/env python3
"""
Test script for the ASR (Automatic Speech Recognition) service.
Tests different speech recognition capabilities and configurations.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
import wave

import numpy as np
import torch

# Import ASR service
from app.services.asr_service import (
    WHISPER_STREAMING_AVAILABLE,
    ASRService,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def set_debug_mode(enable=False):
    """Set debug logging mode."""
    level = logging.DEBUG if enable else logging.INFO
    logger.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)
    # Set debug level for ASR service logger as well
    logging.getLogger("app.services.asr_service").setLevel(level)
    logging.getLogger("whisper_streaming").setLevel(level)


def list_available_models():
    """List available ASR models and backends."""
    logger.info("Checking available ASR backends:")

    if WHISPER_STREAMING_AVAILABLE:
        logger.info("✅ whisper_streaming is available")
    else:
        logger.info("❌ whisper_streaming is NOT available")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(
        f"CUDA {'✅ available' if cuda_available else '❌ NOT available'}"
    )

    # Check for model files
    try:
        from app.config import settings

        logger.info(f"Configured Whisper model: {settings.WHISPER_MODEL}")
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")


def test_asr_streaming(audio_file, source_language="ru", target_language="ru"):
    """Test the ASR service with a pre-recorded audio file using the streaming interface.

    Args:
        audio_file: Path to the audio file to test
        source_language: Source language code
        target_language: Target language code

    Returns:
        Transcription result or None if failed
    """
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return None

    logger.info(f"Testing ASR streaming with file: {audio_file}")
    logger.info(
        f"Source language: {source_language}, Target language: {target_language}"
    )

    # Initialize ASR service with VAD disabled
    try:
        asr_service = ASRService(
            source_language=source_language,
            target_language=target_language,
            use_vad=False,  # Disable VAD
            vad_chunk_size=0.1,  # 100ms chunks
            buffer_trimming="segment",  # Trim on segment boundaries
            buffer_trimming_sec=15.0,  # Trim buffer if longer than 15 seconds
        )
        if not asr_service.asr_model:
            logger.error(
                "Failed to initialize ASR model. Check logs for details."
            )
            return None
    except Exception as e:
        logger.error(f"Error initializing ASR service: {str(e)}")
        logger.error(traceback.format_exc())
        return None

    # Load and prepare audio data
    try:
        import soundfile as sf

        logger.info(f"Loading audio with soundfile: {audio_file}")

        # Load audio data (output is float32 normalized to [-1, 1])
        audio_data, sample_rate = sf.read(audio_file)

        # Ensure mono
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            logger.info(f"Converting {audio_data.shape[1]} channels to mono")
            audio_data = audio_data.mean(axis=1)

        # Resample if needed
        if sample_rate != 16000:
            logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
            import librosa

            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=16000
            )
            sample_rate = 16000

        logger.info(
            f"Audio: {audio_data.shape}, {sample_rate}Hz, range [{audio_data.min():.2f}, {audio_data.max():.2f}]"
        )

        # Convert to int16 for PCM (S16_LE format)
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        logger.info(f"Converted to PCM bytes: {len(audio_bytes)} bytes")
    except Exception as e:
        logger.error(f"Error loading audio: {str(e)}")
        logger.error(traceback.format_exc())
        return None

    # Process audio in chunks
    try:
        # Use 100ms chunks (1600 samples at 16kHz)
        chunk_size = 1600 * 2  # 2 bytes per sample for int16
        start_time = time.time()
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size

        logger.info(f"Processing audio in {total_chunks} chunks...")

        # Store all transcription results
        all_results = []

        for i in range(0, len(audio_bytes), chunk_size):
            chunk_num = i // chunk_size + 1
            chunk = audio_bytes[i : i + chunk_size]

            # Skip empty chunks
            if not chunk:
                logger.debug(f"Skipping empty chunk {chunk_num}")
                continue

            # Insert chunk into ASR service
            asr_service.insert_audio_chunk(chunk)

            # Process intermediate results
            try:
                partial_result = asr_service.process_iter()
                if (
                    partial_result and partial_result[2]
                ):  # Check if there's actual text
                    logger.info(f"Chunk {chunk_num}: {partial_result}")
                    all_results.append(partial_result)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

            # Progress indicator
            if chunk_num % 10 == 0:
                logger.info(
                    f"Processed {chunk_num}/{total_chunks} chunks ({chunk_num / total_chunks * 100:.1f}%)"
                )

            # Small delay to simulate real-time processing
            time.sleep(0.01)

        # Get final result
        logger.info("Processing complete, getting final result...")
        final_result = asr_service.finish()
        processing_time = time.time() - start_time

        # Combine all results
        if all_results:
            # Sort by start time
            all_results.sort(
                key=lambda x: x[0] if x[0] is not None else float("inf")
            )

            # Combine text
            combined_text = " ".join(r[2] for r in all_results if r[2])

            # Get start and end times
            start_time = min(r[0] for r in all_results if r[0] is not None)
            end_time = max(r[1] for r in all_results if r[1] is not None)

            final_result = (start_time, end_time, combined_text)

            logger.info(f"✅ Final transcription: {final_result}")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            return final_result
        else:
            logger.error("❌ No transcription produced")
            return None
    except Exception as e:
        logger.error(f"Error during ASR processing: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description="Test ASR service")
    parser.add_argument(
        "--audio", help="Path to audio file for testing", required=False
    )
    parser.add_argument(
        "--source-lang", default="ru", help="Source language code"
    )
    parser.add_argument(
        "--target-lang", default="ru", help="Target language code"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available ASR models and backends",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed information",
    )

    args = parser.parse_args()

    if args.debug:
        set_debug_mode(True)
        logger.debug("Debug logging enabled")

    if args.list_models:
        list_available_models()
        return

    if not args.audio:
        logger.error("Please provide an audio file path with --audio")
        return

    test_asr_streaming(args.audio, args.source_lang, args.target_lang)


if __name__ == "__main__":
    main()
