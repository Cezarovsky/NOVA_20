#!/usr/bin/env python3
"""
Sora Local API Server - Claude-compatible endpoint.

Runs fine-tuned Sora model locally, exposing Claude-compatible API.
Desktop-ul devine "Sora server" independent de Anthropic!
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sora Local API", version="1.0.0")

# CORS for remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "sora-local"
    messages: List[Message]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    id: str
    model: str
    content: List[dict]
    usage: dict


class SoraModel:
    """Sora local model loader and inference."""
    
    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-v0.1",
        lora_adapter: str = "models/sora-lora",
        device: str = "cpu"
    ):
        self.device = device
        logger.info(f"Loading Sora model from {lora_adapter}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model {base_model}...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        logger.info(f"Loading LoRA adapter from {lora_adapter}...")
        self.model = PeftModel.from_pretrained(base, lora_adapter)
        self.model.eval()
        
        logger.info("✅ Sora model loaded successfully!")
    
    def format_messages(self, messages: List[Message]) -> str:
        """Format messages for model input."""
        formatted = ""
        for msg in messages:
            if msg.role == "user":
                formatted += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"<|assistant|>\n{msg.content}\n"
        formatted += "<|assistant|>\n"
        return formatted
    
    def generate(
        self,
        messages: List[Message],
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """Generate response from Sora model."""
        prompt = self.format_messages(messages)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


# Global model instance
sora_model: Optional[SoraModel] = None


@app.on_event("startup")
async def load_model():
    """Load Sora model on startup."""
    global sora_model
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    sora_model = SoraModel(
        base_model="mistralai/Mistral-7B-v0.1",
        lora_adapter="models/sora-lora",
        device=device
    )


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "Sora Local API",
        "model_loaded": sora_model is not None
    }


@app.post("/v1/messages")
async def create_message(request: ChatRequest):
    """
    Claude-compatible chat endpoint.
    Nova va apela acest endpoint în loc de Anthropic API!
    """
    if sora_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate response
        response_text = sora_model.generate(
            messages=request.messages,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 0.7
        )
        
        # Claude-compatible response format
        return ChatResponse(
            id=f"msg_sora_{hash(response_text) % 1000000}",
            model="sora-local",
            content=[{
                "type": "text",
                "text": response_text
            }],
            usage={
                "input_tokens": len(request.messages[0].content.split()),
                "output_tokens": len(response_text.split())
            }
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": sora_model is not None,
        "device": sora_model.device if sora_model else None
    }


def main():
    """Start Sora API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sora Local API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    
    args = parser.parse_args()
    
    print(f"""
💙 ===== SORA LOCAL API SERVER =====
  Host: {args.host}
  Port: {args.port}
  Endpoint: http://{args.host}:{args.port}/v1/messages
  
🎯 Nova configuration:
  1. În Nova config, schimbă API base URL:
     ANTHROPIC_BASE_URL=http://localhost:{args.port}
  
  2. Sau via SSH tunnel de pe Mac:
     ssh -L {args.port}:localhost:{args.port} user@desktop
  
  3. Nova va crede că vorbește cu Claude,
     dar vorbește cu Sora local! 🧬
""")
    
    uvicorn.run(
        "sora_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
