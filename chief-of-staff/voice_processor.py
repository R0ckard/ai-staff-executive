"""
Voice Processor for AI Chief of Staff
Self-contained voice processing with speech-to-text and text-to-speech capabilities
"""

import os
import io
import base64
import tempfile
from typing import Optional, Dict, Any
import openai
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self):
        """Initialize the voice processor with OpenAI client"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.supported_formats = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']
        
    def process_audio_input(self, audio_data: bytes, format: str = 'webm') -> Dict[str, Any]:
        """
        Process audio input and convert to text using OpenAI Whisper
        
        Args:
            audio_data: Raw audio bytes
            format: Audio format (webm, mp3, wav, etc.)
            
        Returns:
            Dict with transcription result and metadata
        """
        try:
            # Validate format
            if format not in self.supported_formats:
                return {
                    "success": False,
                    "error": f"Unsupported audio format: {format}. Supported: {', '.join(self.supported_formats)}",
                    "text": ""
                }
            
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Use OpenAI Whisper for speech-to-text
                with open(temp_file_path, 'rb') as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                
                # Clean up temporary file
                os.unlink(temp_file_path)
                
                return {
                    "success": True,
                    "text": transcript.strip(),
                    "format": format,
                    "length": len(audio_data)
                }
                
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                raise e
                
        except Exception as e:
            logger.error(f"Error processing audio input: {str(e)}")
            return {
                "success": False,
                "error": f"Audio processing failed: {str(e)}",
                "text": ""
            }
    
    def generate_speech(self, text: str, voice: str = "alloy") -> Dict[str, Any]:
        """
        Generate speech from text using OpenAI TTS
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            
        Returns:
            Dict with audio data and metadata
        """
        try:
            # Validate voice
            valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            if voice not in valid_voices:
                voice = "alloy"  # Default fallback
            
            # Generate speech using OpenAI TTS
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            # Get audio data
            audio_data = response.content
            
            # Encode as base64 for web transmission
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return {
                "success": True,
                "audio_data": audio_base64,
                "format": "mp3",
                "voice": voice,
                "text": text,
                "length": len(audio_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return {
                "success": False,
                "error": f"Speech generation failed: {str(e)}",
                "audio_data": ""
            }
    
    def get_voice_status(self) -> Dict[str, Any]:
        """
        Get current voice processing status and capabilities
        
        Returns:
            Dict with voice system status
        """
        try:
            # Test OpenAI API connectivity
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {
                    "status": "disabled",
                    "reason": "OpenAI API key not configured",
                    "speech_to_text": False,
                    "text_to_speech": False
                }
            
            return {
                "status": "enabled",
                "speech_to_text": True,
                "text_to_speech": True,
                "supported_formats": self.supported_formats,
                "available_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                "model": "whisper-1 + tts-1"
            }
            
        except Exception as e:
            logger.error(f"Error checking voice status: {str(e)}")
            return {
                "status": "error",
                "reason": str(e),
                "speech_to_text": False,
                "text_to_speech": False
            }

# Global voice processor instance
voice_processor = VoiceProcessor()

