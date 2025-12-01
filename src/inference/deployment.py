"""
Model Deployment

Export, serve, and deploy models for production.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

# Optional imports
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class ModelExporter:
    """
    Base class for model export.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize exporter.
        
        Args:
            model: Model to export
        """
        self.model = model
        self.model.eval()
    
    def export(self, path: str, *args, **kwargs):
        """Export model to file."""
        raise NotImplementedError


class ONNXExporter(ModelExporter):
    """
    Export model to ONNX format.
    
    ONNX enables deployment across multiple platforms.
    """
    
    def export(
        self,
        path: str,
        dummy_input: torch.Tensor,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14,
    ):
        """
        Export to ONNX.
        
        Args:
            path: Output path
            dummy_input: Example input for tracing
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
            opset_version: ONNX opset version
        """
        if not HAS_ONNX:
            raise ImportError("ONNX not installed. Run: pip install onnx onnxruntime")
        
        if input_names is None:
            input_names = ['input']
        
        if output_names is None:
            output_names = ['output']
        
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'},
            }
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        print(f"✓ Model exported to ONNX: {path}")
        
        # Verify
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verified")
    
    def load_onnx(self, path: str):
        """
        Load ONNX model for inference.
        
        Args:
            path: Path to ONNX model
            
        Returns:
            ONNX Runtime inference session
        """
        if not HAS_ONNX:
            raise ImportError("ONNX not installed. Run: pip install onnx onnxruntime")
        
        session = ort.InferenceSession(path)
        print(f"✓ ONNX model loaded: {path}")
        return session


class TorchScriptExporter(ModelExporter):
    """
    Export model to TorchScript.
    
    TorchScript enables deployment in C++ environments.
    """
    
    def export(
        self,
        path: str,
        dummy_input: Optional[torch.Tensor] = None,
        method: str = 'trace',
    ):
        """
        Export to TorchScript.
        
        Args:
            path: Output path
            dummy_input: Example input (required for tracing)
            method: Export method ('trace' or 'script')
        """
        if method == 'trace':
            if dummy_input is None:
                raise ValueError("dummy_input required for tracing")
            
            # Trace model
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(path)
            print(f"✓ Model traced and saved: {path}")
        
        elif method == 'script':
            # Script model
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(path)
            print(f"✓ Model scripted and saved: {path}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def load_torchscript(self, path: str) -> torch.jit.ScriptModule:
        """
        Load TorchScript model.
        
        Args:
            path: Path to TorchScript model
            
        Returns:
            Loaded model
        """
        model = torch.jit.load(path)
        print(f"✓ TorchScript model loaded: {path}")
        return model


class ModelServer:
    """
    Simple model serving for inference.
    
    Handles requests and returns predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize model server.
        
        Args:
            model: Model to serve
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def predict(
        self,
        text: str,
        max_length: int = 512,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate prediction for text.
        
        Args:
            text: Input text
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Tokenize
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        # Generate
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.model(generated)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode
        generated_text = self.tokenizer.decode(
            generated[0].tolist(),
            skip_special_tokens=True
        )
        
        return generated_text
    
    def batch_predict(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> List[str]:
        """
        Batch prediction.
        
        Args:
            texts: List of input texts
            max_length: Maximum generation length
            
        Returns:
            List of generated texts
        """
        results = []
        
        for text in texts:
            result = self.predict(text, max_length)
            results.append(result)
        
        return results


class InferenceAPI:
    """
    REST API wrapper for inference.
    
    Provides HTTP endpoints for model serving.
    """
    
    def __init__(
        self,
        server: ModelServer,
        host: str = '0.0.0.0',
        port: int = 8000,
    ):
        """
        Initialize API.
        
        Args:
            server: Model server
            host: Host address
            port: Port number
        """
        self.server = server
        self.host = host
        self.port = port
    
    def create_app(self):
        """
        Create FastAPI application.
        
        Returns:
            FastAPI app
        """
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        except ImportError:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")
        
        app = FastAPI(title="NOVA Inference API")
        
        class GenerateRequest(BaseModel):
            text: str
            max_length: int = 512
            temperature: float = 1.0
        
        class GenerateResponse(BaseModel):
            generated_text: str
        
        @app.post("/generate", response_model=GenerateResponse)
        def generate(request: GenerateRequest):
            """Generate text endpoint."""
            try:
                generated = self.server.predict(
                    request.text,
                    max_length=request.max_length,
                    temperature=request.temperature
                )
                return GenerateResponse(generated_text=generated)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        def health():
            """Health check endpoint."""
            return {"status": "healthy"}
        
        return app
    
    def run(self):
        """
        Run API server.
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError("Uvicorn not installed. Run: pip install uvicorn")
        
        app = self.create_app()
        print(f"Starting API server on {self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)


class ModelPackager:
    """
    Package model with metadata for deployment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Dict[str, Any],
    ):
        """
        Initialize packager.
        
        Args:
            model: Model to package
            tokenizer: Tokenizer
            config: Model configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def package(self, output_dir: str):
        """
        Package model for deployment.
        
        Creates directory with:
        - model.pt: Model weights
        - tokenizer.json: Tokenizer
        - config.json: Configuration
        - metadata.json: Deployment metadata
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Save tokenizer
        tokenizer_path = output_path / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))
        print(f"✓ Tokenizer saved: {tokenizer_path}")
        
        # Save config
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✓ Config saved: {config_path}")
        
        # Save metadata
        metadata = {
            "model_type": "nova",
            "version": "1.0",
            "framework": "pytorch",
            "requires_cuda": torch.cuda.is_available(),
        }
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")
        
        print(f"\n✓ Model package created: {output_dir}")
    
    @staticmethod
    def load_package(package_dir: str, model_class: type) -> tuple:
        """
        Load packaged model.
        
        Args:
            package_dir: Package directory
            model_class: Model class to instantiate
            
        Returns:
            (model, tokenizer, config)
        """
        package_path = Path(package_dir)
        
        # Load config
        config_path = package_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model
        model = model_class(**config)
        model_path = package_path / "model.pt"
        model.load_state_dict(torch.load(model_path))
        print(f"✓ Model loaded: {model_path}")
        
        # Load tokenizer
        from src.data.tokenizer import NovaTokenizer
        tokenizer_path = package_path / "tokenizer.json"
        tokenizer = NovaTokenizer.load(str(tokenizer_path))
        print(f"✓ Tokenizer loaded: {tokenizer_path}")
        
        return model, tokenizer, config
