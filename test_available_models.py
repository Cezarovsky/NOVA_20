import sys
sys.path.insert(0, '/Users/cezartipa/Documents/Nova_20')

from dotenv import load_dotenv
load_dotenv()

import anthropic
import os

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

models_to_test = [
    "claude-3-5-sonnet-20241022",  # Latest Sonnet 3.5
    "claude-3-5-sonnet-20240620",  # Previous Sonnet 3.5
    "claude-3-sonnet-20240229",    # Sonnet 3
    "claude-3-opus-20240229",      # Opus (best quality)
    "claude-3-haiku-20240307",     # Haiku (fastest/cheapest)
]

print("Testing available Claude models...")
print("=" * 70)

working_models = []

for model in models_to_test:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✅ {model:35} - AVAILABLE")
        working_models.append(model)
    except anthropic.NotFoundError:
        print(f"❌ {model:35} - NOT AVAILABLE")
    except Exception as e:
        print(f"⚠️  {model:35} - ERROR: {str(e)[:50]}")

print("\n" + "=" * 70)
print(f"\nWorking models ({len(working_models)}):")
for model in working_models:
    print(f"  • {model}")

if working_models:
    best_model = working_models[0]
    print(f"\n🎯 Recommended: {best_model}")
