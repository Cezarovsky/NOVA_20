"""
🎤 NOVA Voice Demo

Demonstrate NOVA's text-to-speech capabilities with interactive examples.
"""

import torch
from pathlib import Path
import time

# Import NOVA components
from src.ml.transformer import Transformer
from src.data.tokenizer import NovaTokenizer
from src.voice import NovaVoice


def print_banner():
    """Print NOVA voice banner."""
    banner = """
    ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗     🎤
    ████╗  ██║██╔═══██╗██║   ██║██╔══██╗    
    ██╔██╗ ██║██║   ██║██║   ██║███████║    
    ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║    
    ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║    
    ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝    
    
    Voice-Enabled Virtual Assistant 🗣️✨
    """
    print(banner)


def demo_basic_speech():
    """Demo 1: Basic speech with different rates."""
    print("\n" + "="*60)
    print("🎤 Demo 1: Basic Speech")
    print("="*60)
    
    voice = NovaVoice()
    
    messages = [
        "Hello! I am NOVA, your Neural Optimized Virtual Assistant.",
        "I can speak with you using text to speech.",
        "My voice works completely offline, without internet connection."
    ]
    
    for msg in messages:
        print(f"\n💬 NOVA: {msg}")
        voice.speak(msg)
        time.sleep(0.3)


def demo_voice_customization():
    """Demo 2: Voice customization with different rates and volumes."""
    print("\n" + "="*60)
    print("🎨 Demo 2: Voice Customization")
    print("="*60)
    
    # Slow and quiet
    print("\n📢 Slow and gentle (rate=120, volume=0.6):")
    voice_slow = NovaVoice(rate=120, volume=0.6)
    print("💬 NOVA: I can speak slowly and quietly.")
    voice_slow.speak("I can speak slowly and quietly.")
    
    time.sleep(0.5)
    
    # Fast and loud
    print("\n📢 Fast and energetic (rate=220, volume=1.0):")
    voice_fast = NovaVoice(rate=220, volume=1.0)
    print("💬 NOVA: Or I can speak quickly and with more energy!")
    voice_fast.speak("Or I can speak quickly and with more energy!")
    
    time.sleep(0.5)
    
    # Normal
    print("\n📢 Normal voice (rate=175, volume=0.9):")
    voice_normal = NovaVoice()
    print("💬 NOVA: But I prefer this comfortable speaking pace.")
    voice_normal.speak("But I prefer this comfortable speaking pace.")


def demo_multilingual():
    """Demo 3: Multilingual speech (Romanian and English)."""
    print("\n" + "="*60)
    print("🌍 Demo 3: Multilingual Speech")
    print("="*60)
    
    voice = NovaVoice()
    
    messages = [
        ("en", "Hello! I can speak in English."),
        ("ro", "Bună! Pot vorbi și în limba română."),
        ("en", "I support multiple languages."),
        ("ro", "Suport mai multe limbi."),
        ("en", "Let's work together!"),
        ("ro", "Hai să lucrăm împreună!")
    ]
    
    for lang, msg in messages:
        flag = "🇬🇧" if lang == "en" else "🇷🇴"
        print(f"\n{flag} NOVA: {msg}")
        voice.speak(msg)
        time.sleep(0.3)


def demo_with_generation():
    """Demo 4: Text generation with voice output."""
    print("\n" + "="*60)
    print("🤖 Demo 4: Generation with Voice")
    print("="*60)
    
    # Setup
    device = torch.device('cpu')
    print(f"\n⚙️  Device: {device}")
    
    # Create small model
    print("\n🤖 Creating NOVA model...")
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_len=100,
        dropout=0.1
    ).to(device)
    print("✓ Model ready")
    
    # Create tokenizer
    print("\n📝 Training tokenizer...")
    tokenizer = NovaTokenizer(vocab_size=1000)
    corpus = [
        "Hello, how are you today?",
        "I am doing great, thank you!",
        "NOVA is a smart assistant.",
        "Let me help you with that."
    ]
    tokenizer.train(corpus)
    print(f"✓ Tokenizer ready with {len(tokenizer.token_to_id)} tokens")
    
    # Create voice
    voice = NovaVoice()
    
    # Test input
    test_text = "Hello"
    print(f"\n💭 Input: '{test_text}'")
    voice.speak(f"Processing input: {test_text}")
    
    # Tokenize and forward pass
    input_ids = torch.tensor([tokenizer.encode(test_text)]).to(device)
    
    with torch.no_grad():
        output = model(input_ids, input_ids)
    
    # Get prediction
    probs = torch.softmax(output[0, -1], dim=-1)
    top_token_id = probs.argmax().item()
    predicted_token = tokenizer.id_to_token.get(top_token_id, '<unk>')
    
    result = f"My prediction is: {predicted_token}"
    print(f"🎯 {result}")
    voice.speak(result)
    
    time.sleep(0.5)
    voice.speak("This shows how I can generate text and speak it to you!")


def demo_list_voices():
    """Demo 5: List all available voices."""
    print("\n" + "="*60)
    print("📋 Demo 5: Available Voices")
    print("="*60)
    
    voice = NovaVoice()
    voices = voice.get_available_voices()
    
    print(f"\nFound {len(voices)} voices on your system:")
    for i, v in enumerate(voices[:5], 1):  # Show first 5
        print(f"\n{i}. {v['name']}")
        if v['languages']:
            print(f"   Languages: {', '.join(v['languages'])}")
    
    if len(voices) > 5:
        print(f"\n... and {len(voices) - 5} more voices available")


def main():
    """Run all voice demos."""
    print_banner()
    
    print("\n" + "="*60)
    print("🚀 Starting NOVA Voice Demos")
    print("="*60)
    
    try:
        # Demo 1: Basic speech
        demo_basic_speech()
        
        time.sleep(1)
        
        # Demo 2: Voice customization
        demo_voice_customization()
        
        time.sleep(1)
        
        # Demo 3: Multilingual
        demo_multilingual()
        
        time.sleep(1)
        
        # Demo 4: Generation with voice
        demo_with_generation()
        
        time.sleep(1)
        
        # Demo 5: List voices
        demo_list_voices()
        
        # Final message
        print("\n" + "="*60)
        print("✅ All Voice Demos Complete!")
        print("="*60)
        
        voice = NovaVoice()
        voice.speak("All voice demonstrations are complete! I am ready to assist you.")
        
        print("\n💙 NOVA - Now with voice! ✨🎤")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
