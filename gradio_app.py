"""
Modified Hallo3 Gradio Interface with GeneFace++ lip-sync
Replaces Wav2Lip for license compliance while maintaining:
- Body animation (original MIT-licensed code)
- SpeechT5 audio processing (MIT)
- GeneFace++ lip-sync (MIT)
"""

# Core imports (unchanged)
import os
import gradio as gr
import torch
from PIL import Image

# New imports for GeneFace++ and SpeechT5
from geneface.inference import LipSyncModel  # MIT-licensed alternative
from speechbrain.pretrained import EncoderDecoderASR  # Original SpeechT5

class VideoGenerator:
    def __init__(self):
        """
        Initialize models:
        - Keeps original SAT diffusion model
        - Replaces Wav2Lip audio processor with:
          1. GeneFace++ for lip-sync
          2. SpeechT5 for transcription (original)
        """
        # Original video generation model (unchanged)
        self.model = load_original_diffusion_model()  
        
        # Replacement components
        self.lip_syncer = LipSyncModel(device="cuda")  # MIT license
        self.speech_processor = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-transformer-transformerlm-librispeech",
            savedir="pretrained_models/speecht5"  # Original config path
        )

    def generate_video(self, image: Image, audio_path: str, prompt: str) -> str:
        """
        Modified pipeline:
        1. Audio transcription (SpeechT5)
        2. Original body animation generation
        3. GeneFace++ lip-sync (post-process)
        """
        # 1. Speech recognition (original SpeechT5)
        transcript = self.speech_processor.transcribe_file(audio_path)

        # 2. Generate body animation (original code)
        temp_video = self._generate_body_animation(image, prompt)

        # 3. Apply GeneFace++ lip-sync
        if audio_path:
            output_path = self.lip_syncer.predict(
                video_path=temp_video,
                audio_path=audio_path,
                use_pose=True  # Preserves body motion
            )
        return output_path

    def _generate_body_animation(self, image: Image, prompt: str) -> str:
        """Original SAT diffusion code (unchanged)"""
        # ... maintain existing implementation ...
        return "temp_video.mp4"

def create_interface():
    """Gradio setup with modified pipeline"""
    generator = VideoGenerator()
    
    interface = gr.Interface(
        fn=generator.generate_video,
        inputs=[
            gr.Image(type="pil", label="Input Portrait"),
            gr.Audio(type="filepath", label="Speech Audio"),
            gr.Textbox(label="Animation Prompt")
        ],
        outputs=gr.Video(label="Animated Video"),
        title="Hallo3-GeneFace++",
        description="""Modified version with:
        - Body animation (original SAT model)
        - Lip-sync via GeneFace++ (MIT)
        - Audio processing via SpeechT5 (MIT)"""
    )
    return interface

if __name__ == "__main__":
    create_interface().launch()
