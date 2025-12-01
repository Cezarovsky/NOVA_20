"""
🎤 NOVA Text-to-Speech Module

Provides offline text-to-speech capabilities using pyttsx3 with feminine voice.
Works completely offline without internet connection.
"""

import pyttsx3
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class NovaVoice:
    """
    Text-to-speech engine for NOVA with feminine voice configuration.
    
    Features:
    - Offline TTS using pyttsx3
    - Feminine voice selection
    - Adjustable speech rate and volume
    - Multiple voice options with fallback
    """
    
    def __init__(
        self,
        rate: int = 175,  # Speech rate (words per minute)
        volume: float = 0.9,  # Volume (0.0 to 1.0)
        voice_preference: str = "feminine"
    ):
        """
        Initialize NOVA's voice.
        
        Args:
            rate: Speech rate in words per minute (default: 175)
            volume: Volume level from 0.0 to 1.0 (default: 0.9)
            voice_preference: "feminine" or "masculine" (default: "feminine")
        """
        self.engine = pyttsx3.init()
        self.rate = rate
        self.volume = volume
        self.voice_preference = voice_preference
        
        # Configure voice
        self._setup_voice()
        self._configure_engine()
        
        logger.info(f"✓ NovaVoice initialized with {voice_preference} voice")
    
    def _setup_voice(self):
        """Select the best feminine voice available on the system."""
        voices = self.engine.getProperty('voices')
        
        # Try to find a feminine voice
        feminine_voice = None
        for voice in voices:
            voice_name = voice.name.lower()
            
            # On macOS, look for common feminine voices
            if any(name in voice_name for name in ['samantha', 'victoria', 'karen', 'moira', 'fiona']):
                feminine_voice = voice.id
                logger.info(f"Selected voice: {voice.name}")
                break
            
            # Generic check for female indicators
            if 'female' in voice_name or 'woman' in voice_name:
                feminine_voice = voice.id
                logger.info(f"Selected voice: {voice.name}")
                break
        
        # Set the voice or use default
        if feminine_voice:
            self.engine.setProperty('voice', feminine_voice)
        else:
            # Use first available voice as fallback
            if voices:
                self.engine.setProperty('voice', voices[0].id)
                logger.warning(f"Using default voice: {voices[0].name}")
    
    def _configure_engine(self):
        """Configure speech rate and volume."""
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)
    
    def speak(self, text: str, wait: bool = True):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            wait: If True, wait until speech is finished. If False, return immediately.
        """
        try:
            self.engine.say(text)
            if wait:
                self.engine.runAndWait()
            else:
                # For async speech, engine runs in background
                self.engine.startLoop(False)
                self.engine.iterate()
                self.engine.endLoop()
        except Exception as e:
            logger.error(f"Error speaking: {e}")
    
    def save_to_file(self, text: str, filename: str):
        """
        Save speech to an audio file instead of playing it.
        
        Args:
            text: Text to convert to speech
            filename: Output audio file path
        """
        try:
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            logger.info(f"✓ Saved speech to {filename}")
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
    
    def set_rate(self, rate: int):
        """
        Change speech rate.
        
        Args:
            rate: Words per minute (typical range: 100-300)
        """
        self.rate = rate
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """
        Change volume level.
        
        Args:
            volume: Volume from 0.0 (silent) to 1.0 (maximum)
        """
        self.volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
        self.engine.setProperty('volume', self.volume)
    
    def get_available_voices(self) -> List[dict]:
        """
        Get list of all available voices on the system.
        
        Returns:
            List of voice information dictionaries
        """
        voices = self.engine.getProperty('voices')
        return [
            {
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender if hasattr(voice, 'gender') else 'unknown',
                'age': voice.age if hasattr(voice, 'age') else 'unknown'
            }
            for voice in voices
        ]
    
    def list_voices(self):
        """Print all available voices in a readable format."""
        voices = self.get_available_voices()
        print(f"\n📢 Available voices ({len(voices)}):")
        print("=" * 60)
        for i, voice in enumerate(voices, 1):
            print(f"{i}. {voice['name']}")
            print(f"   ID: {voice['id']}")
            if voice['languages']:
                print(f"   Languages: {voice['languages']}")
            print()
    
    def stop(self):
        """Stop current speech."""
        try:
            self.engine.stop()
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'engine'):
                self.engine.stop()
        except:
            pass


# Convenience function for quick use
def speak(text: str, rate: int = 175, volume: float = 0.9):
    """
    Quick function to speak text without creating a NovaVoice object.
    
    Args:
        text: Text to speak
        rate: Speech rate in words per minute
        volume: Volume level (0.0 to 1.0)
    """
    voice = NovaVoice(rate=rate, volume=volume)
    voice.speak(text)


if __name__ == "__main__":
    # Demo
    print("🎤 NOVA Voice Demo\n")
    
    # Create voice
    voice = NovaVoice()
    
    # List available voices
    voice.list_voices()
    
    # Test speech
    print("Testing NOVA's voice...")
    voice.speak("Hello! I am NOVA, your Neural Optimized Virtual Assistant.")
    voice.speak("I can speak offline without internet connection.")
    voice.speak("How can I assist you today?")
    
    print("\n✓ Voice demo complete!")
