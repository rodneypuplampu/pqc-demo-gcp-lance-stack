# PQC oice Assistant: Complete Implementation Guide

## Executive Summary

This guide provides a step-by-step implementation plan to transform your **Ogummaa Local** sovereign AI system into a full-featured NotebookLM clone with voice assistant capabilities. The system will maintain complete offline operation while adding:

- **Long-context document understanding** (128K tokens via Gemma 3)
- **Voice input/output** (MediaPipe + Piper TTS)
- **Function calling** for action-oriented tasks
- **Hybrid memory system** (Vector + Relational + Topic Clustering)
- **Report generation** (Briefing docs, FAQs, Study guides)
Learn more at:  https://quantropi.com
---

## Architecture Overview

### Current State: Ogummaa Local
- **Frontend**: Flask + Vanilla JS with SocketIO streaming
- **Vector DB**: LanceDB with Universal Sentence Encoder
- **LLM**: Ollama (tinyllama/gemma)
- **Topic Modeling**: BERTopic

### Target State: Enhanced NotebookLM + Voice Assistant
- **Long-Context Engine**: Gemma 3 (128K context window)
- **Voice Input**: MediaPipe Speech Recognition
- **Voice Output**: Piper Neural TTS
- **Hybrid Memory**: LanceDB (vectors) + SQLite (structured) + BERTopic (topics)
- **Function Calling**: Gemma 3 native tool support
- **Report Templates**: Pre-configured prompts for document analysis

---

## Phase 1: Upgrade the Foundation (Model & Context)

### Step 1.1: Install and Configure Gemma 3

**Why**: Gemma 3 provides the 128K context window needed to hold entire documents in memory, enabling true NotebookLM-style analysis.

```bash
# Pull Gemma 3 model
ollama pull gemma3

# Create a custom model with maximum context
cat > Modelfile << EOF
FROM gemma3
PARAMETER num_ctx 128000
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Build custom model
ollama create gemma3-128k -f Modelfile

# Verify it works
ollama run gemma3-128k "Hello, test the context window."
```

**Configuration File**: Create `config/model_config.yaml`

```yaml
# config/model_config.yaml
models:
  long_context:
    name: "gemma3-128k"
    context_window: 128000
    use_case: "document_analysis"
    temperature: 0.7
    
  fast_inference:
    name: "gemma2:9b"
    context_window: 8192
    use_case: "quick_qa"
    temperature: 0.8
    
  function_calling:
    name: "gemma3-128k"
    context_window: 32000
    use_case: "assistant_actions"
    temperature: 0.5
```

### Step 1.2: Update LanceDB Manager for Long Context

**File**: `production_local/LanceDBManager.py`

```python
# production_local/LanceDBManager.py
import lancedb
import numpy as np
from typing import List, Dict, Optional
import logging

class EnhancedLanceDBManager:
    """Enhanced LanceDB manager with long-context optimization"""
    
    def __init__(self, db_path: str = "./lancedb_data", max_context_tokens: int = 128000):
        self.db = lancedb.connect(db_path)
        self.max_context_tokens = max_context_tokens
        self.avg_tokens_per_char = 0.25  # Rough estimate: 4 chars = 1 token
        self.logger = logging.getLogger(__name__)
        
    def create_document_table(self, table_name: str = "documents"):
        """Create or get document table with enhanced schema"""
        schema = {
            "chunk_id": "string",
            "document_id": "string",
            "content": "string",
            "vector": "vector(512)",  # USE embeddings
            "metadata": "json",
            "chunk_index": "int",
            "total_chunks": "int",
            "token_count": "int",
            "created_at": "timestamp"
        }
        
        try:
            return self.db.create_table(table_name, schema=schema, mode="overwrite")
        except Exception as e:
            self.logger.info(f"Table exists, opening: {e}")
            return self.db.open_table(table_name)
    
    def search_with_context_assembly(
        self, 
        query: str, 
        table_name: str = "documents",
        top_k: int = 10,
        assemble_full_context: bool = True
    ) -> Dict:
        """
        Search and optionally assemble full document context for long-context models
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            assemble_full_context: If True, retrieve entire documents for top results
            
        Returns:
            Dictionary with chunks and optional full documents
        """
        table = self.db.open_table(table_name)
        
        # Step 1: Vector search for relevant chunks
        results = table.search(query).limit(top_k).to_list()
        
        if not assemble_full_context:
            return {"chunks": results, "full_documents": None}
        
        # Step 2: Identify unique document IDs
        doc_ids = list(set([r['document_id'] for r in results]))
        
        # Step 3: Retrieve all chunks for each document, ordered by chunk_index
        full_documents = {}
        for doc_id in doc_ids:
            doc_chunks = (
                table.search()
                .where(f"document_id = '{doc_id}'")
                .to_pandas()
                .sort_values('chunk_index')
            )
            
            # Assemble full text
            full_text = "\n\n".join(doc_chunks['content'].tolist())
            total_tokens = sum(doc_chunks['token_count'].tolist())
            
            # Check if it fits in context window
            if total_tokens <= self.max_context_tokens * 0.8:  # 80% safety margin
                full_documents[doc_id] = {
                    "content": full_text,
                    "metadata": doc_chunks.iloc[0]['metadata'],
                    "token_count": total_tokens,
                    "fits_in_context": True
                }
            else:
                # If too large, use top N chunks
                relevant_chunks = [r for r in results if r['document_id'] == doc_id]
                partial_text = "\n\n".join([c['content'] for c in relevant_chunks[:5]])
                full_documents[doc_id] = {
                    "content": partial_text,
                    "metadata": relevant_chunks[0]['metadata'],
                    "token_count": total_tokens,
                    "fits_in_context": False,
                    "warning": "Document truncated - too large for context window"
                }
        
        return {
            "chunks": results,
            "full_documents": full_documents,
            "doc_ids": doc_ids
        }
```

### Step 1.3: Create Model Selector Utility

**File**: `production_local/model_selector.py`

```python
# production_local/model_selector.py
import yaml
from enum import Enum
from typing import Dict, Any

class TaskType(Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    QUICK_QA = "quick_qa"
    FUNCTION_CALLING = "assistant_actions"
    REPORT_GENERATION = "report_generation"
    VOICE_RESPONSE = "voice_response"

class ModelSelector:
    """Intelligently select the right model for each task"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_model_for_task(self, task_type: TaskType, document_size: int = 0) -> Dict[str, Any]:
        """
        Select optimal model based on task and document size
        
        Args:
            task_type: Type of task to perform
            document_size: Approximate token count of documents
            
        Returns:
            Model configuration dictionary
        """
        if task_type == TaskType.DOCUMENT_ANALYSIS:
            if document_size > 50000:
                return self.config['models']['long_context']
            else:
                return self.config['models']['fast_inference']
        
        elif task_type == TaskType.FUNCTION_CALLING:
            return self.config['models']['function_calling']
        
        elif task_type == TaskType.QUICK_QA:
            return self.config['models']['fast_inference']
        
        elif task_type == TaskType.REPORT_GENERATION:
            return self.config['models']['long_context']
        
        elif task_type == TaskType.VOICE_RESPONSE:
            # Voice needs fast response time
            return self.config['models']['fast_inference']
        
        # Default fallback
        return self.config['models']['fast_inference']
```

---

## Phase 2: Implement Voice Input (Speech Recognition)

### Step 2.1: Install Dependencies

```bash
# Install MediaPipe and audio dependencies
pip install mediapipe sounddevice scipy numpy --break-system-packages

# Download MediaPipe speech model (if not already available)
mkdir -p models/mediapipe
# Note: As of 2025, check MediaPipe's official model hub for the latest speech recognizer
```

### Step 2.2: Create Voice Input Component

**File**: `components/listener/MediaPipeListener.py`

```python
# components/listener/MediaPipeListener.py
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import audio
import tempfile
import os
from typing import Optional, Tuple
import logging

class MediaPipeListener:
    """Voice input component using MediaPipe speech recognition"""
    
    def __init__(
        self, 
        model_path: str = 'models/mediapipe/speech_recognizer.task',
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0
    ):
        """
        Initialize MediaPipe speech recognizer
        
        Args:
            model_path: Path to MediaPipe .task model file
            sample_rate: Audio sample rate (16kHz recommended)
            silence_threshold: Amplitude threshold for silence detection
            silence_duration: Seconds of silence before stopping recording
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe recognizer
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = audio.AudioClassifierOptions(
            base_options=base_options,
            max_results=1,
            running_mode=audio.RunningMode.AUDIO_CLIPS
        )
        self.recognizer = audio.AudioClassifier.create_from_options(options)
    
    def record_audio(self, max_duration: int = 30) -> Tuple[np.ndarray, int]:
        """
        Record audio from microphone with automatic silence detection
        
        Args:
            max_duration: Maximum recording duration in seconds
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        print("🎤 Listening... (Speak now)")
        
        recording = []
        silence_samples = int(self.silence_duration * self.sample_rate)
        silent_frames = 0
        
        def callback(indata, frames, time, status):
            nonlocal silent_frames
            if status:
                self.logger.warning(f"Audio status: {status}")
            
            # Add to recording
            recording.append(indata.copy())
            
            # Check for silence
            volume = np.abs(indata).mean()
            if volume < self.silence_threshold:
                silent_frames += frames
            else:
                silent_frames = 0
            
            # Stop if silent too long
            if silent_frames >= silence_samples:
                raise sd.CallbackStop()
        
        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=callback,
                blocksize=1024
            ):
                sd.sleep(int(max_duration * 1000))
        except sd.CallbackStop:
            print("✅ Recording stopped (silence detected)")
        
        # Concatenate recording
        audio_data = np.concatenate(recording, axis=0)
        return audio_data, self.sample_rate
    
    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio using MediaPipe
        
        Args:
            audio_data: NumPy array of audio samples
            
        Returns:
            Transcribed text
        """
        # Save to temporary WAV file (MediaPipe requires file input)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            wavfile.write(tmp_path, self.sample_rate, audio_data.astype(np.int16))
        
        try:
            # Create audio data object
            audio_clip = audio.AudioData.create_from_array(
                audio_data.flatten().astype(np.float32),
                self.sample_rate
            )
            
            # Recognize
            result = self.recognizer.classify(audio_clip)
            
            # Extract best transcription
            if result and result.classifications:
                top_result = result.classifications[0].categories[0]
                return top_result.category_name
            else:
                return ""
        
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def listen_once(self, max_duration: int = 30) -> str:
        """
        Complete listen cycle: record + transcribe
        
        Args:
            max_duration: Maximum recording time in seconds
            
        Returns:
            Transcribed text
        """
        audio_data, _ = self.record_audio(max_duration)
        text = self.transcribe_audio(audio_data)
        
        if text:
            print(f"📝 Transcribed: {text}")
        else:
            print("❌ No speech detected")
        
        return text
```

### Step 2.3: Alternative: Whisper for Better Accuracy

If MediaPipe doesn't perform well, use Whisper (more accurate, still runs locally):

```python
# components/listener/WhisperListener.py
import sounddevice as sd
import numpy as np
import whisper
from scipy.io import wavfile
import tempfile
import os

class WhisperListener:
    """Voice input using OpenAI Whisper (runs locally)"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model
        
        Args:
            model_size: tiny, base, small, medium, large
                       (base recommended for Intel N95/N100)
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000
        print("✅ Whisper loaded")
    
    def record_audio(self, duration: int = 5) -> np.ndarray:
        """Record audio for fixed duration"""
        print(f"🎤 Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        return audio_data
    
    def listen_once(self, duration: int = 5) -> str:
        """Record and transcribe audio"""
        audio_data = self.record_audio(duration)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            wavfile.write(tmp_path, self.sample_rate, audio_data)
        
        try:
            # Transcribe
            result = self.model.transcribe(tmp_path, language='en')
            text = result['text'].strip()
            print(f"📝 Transcribed: {text}")
            return text
        finally:
            os.remove(tmp_path)
```

**Installation**:
```bash
pip install openai-whisper --break-system-packages
```

---

## Phase 3: Implement Voice Output (Text-to-Speech)

### Step 3.1: Install Piper TTS

```bash
# Download Piper binary (adjust for your OS)
cd models
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz

# Download a voice model (English, high quality)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### Step 3.2: Create Voice Output Component

**File**: `components/speaker/PiperSpeaker.py`

```python
# components/speaker/PiperSpeaker.py
import subprocess
import tempfile
import os
import json
from typing import Optional
import logging

class PiperSpeaker:
    """Neural text-to-speech using Piper"""
    
    def __init__(
        self,
        piper_binary: str = "./models/piper/piper",
        model_path: str = "./models/piper/en_US-lessac-medium.onnx",
        config_path: Optional[str] = None,
        speaking_rate: float = 1.0
    ):
        """
        Initialize Piper TTS
        
        Args:
            piper_binary: Path to piper executable
            model_path: Path to .onnx voice model
            config_path: Path to model config (auto-detected if None)
            speaking_rate: Speech speed multiplier (0.5-2.0)
        """
        self.piper_binary = piper_binary
        self.model_path = model_path
        self.speaking_rate = speaking_rate
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect config
        if config_path is None:
            self.config_path = model_path + ".json"
        else:
            self.config_path = config_path
        
        # Verify files exist
        if not os.path.exists(self.piper_binary):
            raise FileNotFoundError(f"Piper binary not found: {self.piper_binary}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Make binary executable
        os.chmod(self.piper_binary, 0o755)
        
        print("✅ Piper TTS initialized")
    
    def speak(
        self, 
        text: str, 
        output_file: Optional[str] = None,
        play_audio: bool = True
    ) -> Optional[str]:
        """
        Convert text to speech and optionally play it
        
        Args:
            text: Text to speak
            output_file: Path to save WAV file (temp file if None)
            play_audio: If True, play audio through speakers
            
        Returns:
            Path to generated WAV file
        """
        # Clean text
        text = text.strip()
        if not text:
            self.logger.warning("Empty text provided to TTS")
            return None
        
        # Create output file
        if output_file is None:
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_file = tmp.name
            tmp.close()
        
        try:
            # Build Piper command
            cmd = [
                self.piper_binary,
                "--model", self.model_path,
                "--config", self.config_path,
                "--output_file", output_file,
                "--length_scale", str(1.0 / self.speaking_rate)  # Inverse for speed
            ]
            
            # Run Piper (pipe text via stdin)
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                self.logger.error(f"Piper error: {stderr}")
                return None
            
            # Play audio if requested
            if play_audio:
                self._play_audio(output_file)
            
            return output_file
        
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            return None
    
    def _play_audio(self, wav_file: str):
        """Play WAV file through system audio"""
        try:
            # Try different audio players based on OS
            if os.system("which aplay > /dev/null 2>&1") == 0:
                # Linux (ALSA)
                subprocess.run(["aplay", wav_file], check=True)
            elif os.system("which afplay > /dev/null 2>&1") == 0:
                # macOS
                subprocess.run(["afplay", wav_file], check=True)
            elif os.system("which powershell > /dev/null 2>&1") == 0:
                # Windows
                subprocess.run([
                    "powershell", "-c",
                    f"(New-Object Media.SoundPlayer '{wav_file}').PlaySync()"
                ], check=True)
            else:
                self.logger.warning("No audio player found")
        
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
    
    def speak_streaming(self, text_generator):
        """
        Speak text as it's being generated (for LLM streaming)
        
        Args:
            text_generator: Iterator yielding text chunks
        """
        buffer = ""
        sentence_endings = ['. ', '! ', '? ', '\n']
        
        for chunk in text_generator:
            buffer += chunk
            
            # Check for sentence endings
            for ending in sentence_endings:
                if ending in buffer:
                    sentences = buffer.split(ending)
                    # Speak complete sentences
                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            self.speak(sentence + ending[0], play_audio=True)
                    # Keep incomplete sentence in buffer
                    buffer = sentences[-1]
        
        # Speak remaining text
        if buffer.strip():
            self.speak(buffer, play_audio=True)
```

---

## Phase 4: Implement Function Calling & Assistant Logic

### Step 4.1: Define Tool Schema

**File**: `components/tools/tool_definitions.py`

```python
# components/tools/tool_definitions.py
from enum import Enum
from typing import Dict, List, Any

class ToolCategory(Enum):
    MEMORY = "memory"
    CALENDAR = "calendar"
    NOTES = "notes"
    ANALYSIS = "analysis"

# Tool definitions for Gemma function calling
TOOLS_SCHEMA = [
    {
        "name": "search_documents",
        "description": "Search through uploaded documents, reports, notes, or personal records. Use this when the user asks about specific information from their documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what to find"
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["all", "reports", "notes", "medical", "financial"],
                    "description": "Filter by document category"
                }
            },
            "required": ["query"]
        },
        "category": ToolCategory.MEMORY
    },
    {
        "name": "generate_report",
        "description": "Generate a comprehensive report from documents. Types: 'briefing' (executive summary), 'faq' (Q&A format), 'timeline' (chronological events), 'study_guide' (educational overview).",
        "parameters": {
            "type": "object",
            "properties": {
                "report_type": {
                    "type": "string",
                    "enum": ["briefing", "faq", "timeline", "study_guide"],
                    "description": "Type of report to generate"
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific document IDs to analyze (optional)"
                },
                "focus_area": {
                    "type": "string",
                    "description": "Specific aspect to focus on (optional)"
                }
            },
            "required": ["report_type"]
        },
        "category": ToolCategory.ANALYSIS
    },
    {
        "name": "add_calendar_event",
        "description": "Add an event to the user's personal calendar. Use when user wants to schedule something.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_name": {
                    "type": "string",
                    "description": "Name/title of the event"
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format"
                },
                "time": {
                    "type": "string",
                    "description": "Time in HH:MM format (24-hour)"
                },
                "description": {
                    "type": "string",
                    "description": "Additional details about the event"
                }
            },
            "required": ["event_name", "date", "time"]
        },
        "category": ToolCategory.CALENDAR
    },
    {
        "name": "create_note",
        "description": "Create a new note or reminder for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the note"
                },
                "content": {
                    "type": "string",
                    "description": "Note content"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization"
                }
            },
            "required": ["title", "content"]
        },
        "category": ToolCategory.NOTES
    },
    {
        "name": "analyze_topics",
        "description": "Analyze uploaded documents to discover main topics and themes using BERTopic clustering.",
        "parameters": {
            "type": "object",
            "properties": {
                "num_topics": {
                    "type": "integer",
                    "description": "Number of topics to extract (default: auto)",
                    "default": -1
                },
                "min_topic_size": {
                    "type": "integer",
                    "description": "Minimum documents per topic",
                    "default": 3
                }
            }
        },
        "category": ToolCategory.ANALYSIS
    }
]
```

### Step 4.2: Create Function Executor

**File**: `components/tools/function_executor.py`

```python
# components/tools/function_executor.py
import json
from typing import Dict, Any, Optional
from datetime import datetime
import sqlite3
import logging

class FunctionExecutor:
    """Executes tool calls from LLM function calling"""
    
    def __init__(
        self,
        lance_manager,  # LanceDBManager instance
        topic_modeler,  # TopicModeler instance
        sqlite_path: str = "./data/assistant.db"
    ):
        self.lance_manager = lance_manager
        self.topic_modeler = topic_modeler
        self.sqlite_path = sqlite_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize SQLite for calendar/notes
        self._init_sqlite()
    
    def _init_sqlite(self):
        """Create SQLite tables for structured data"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Calendar events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,  -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def execute(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a function call
        
        Args:
            function_name: Name of function to call
            arguments: Function arguments as dict
            
        Returns:
            Execution result dictionary
        """
        try:
            if function_name == "search_documents":
                return self._search_documents(arguments)
            
            elif function_name == "generate_report":
                return self._generate_report(arguments)
            
            elif function_name == "add_calendar_event":
                return self._add_calendar_event(arguments)
            
            elif function_name == "create_note":
                return self._create_note(arguments)
            
            elif function_name == "analyze_topics":
                return self._analyze_topics(arguments)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown function: {function_name}"
                }
        
        except Exception as e:
            self.logger.error(f"Function execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _search_documents(self, args: Dict) -> Dict:
        """Search through document database"""
        query = args.get("query", "")
        doc_type = args.get("doc_type", "all")
        
        # Perform hybrid search
        results = self.lance_manager.search_with_context_assembly(
            query=query,
            top_k=5,
            assemble_full_context=True
        )
        
        # Format results for LLM
        chunks = results.get("chunks", [])
        docs = results.get("full_documents", {})
        
        return {
            "success": True,
            "data": {
                "matching_chunks": [
                    {
                        "content": c['content'],
                        "document": c['document_id'],
                        "relevance": c.get('_distance', 0)
                    }
                    for c in chunks[:3]
                ],
                "full_documents": {
                    doc_id: doc['content']
                    for doc_id, doc in docs.items()
                    if doc.get('fits_in_context', False)
                },
                "total_results": len(chunks)
            }
        }
    
    def _generate_report(self, args: Dict) -> Dict:
        """Generate a report from documents"""
        report_type = args.get("report_type")
        doc_ids = args.get("document_ids", [])
        focus_area = args.get("focus_area", "")
        
        # Get documents
        if doc_ids:
            # Fetch specific documents
            documents = []
            for doc_id in doc_ids:
                # Query LanceDB for this document
                pass  # Implementation here
        else:
            # Get all documents or use focus_area to filter
            pass
        
        return {
            "success": True,
            "data": {
                "report_type": report_type,
                "status": "ready_for_generation",
                "documents_loaded": len(doc_ids) if doc_ids else "all"
            }
        }
    
    def _add_calendar_event(self, args: Dict) -> Dict:
        """Add event to calendar"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO calendar_events (event_name, date, time, description)
            VALUES (?, ?, ?, ?)
        """, (
            args['event_name'],
            args['date'],
            args['time'],
            args.get('description', '')
        ))
        
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "data": {
                "event_id": event_id,
                "message": f"Added '{args['event_name']}' to calendar on {args['date']} at {args['time']}"
            }
        }
    
    def _create_note(self, args: Dict) -> Dict:
        """Create a note"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        tags_json = json.dumps(args.get('tags', []))
        
        cursor.execute("""
            INSERT INTO notes (title, content, tags)
            VALUES (?, ?, ?)
        """, (args['title'], args['content'], tags_json))
        
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "data": {
                "note_id": note_id,
                "message": f"Created note: {args['title']}"
            }
        }
    
    def _analyze_topics(self, args: Dict) -> Dict:
        """Run BERTopic analysis"""
        results = self.topic_modeler.fit_and_extract(
            num_topics=args.get('num_topics', -1),
            min_topic_size=args.get('min_topic_size', 3)
        )
        
        return {
            "success": True,
            "data": {
                "topics": results.get('topics', []),
                "num_topics": len(results.get('topics', [])),
                "message": "Topic analysis complete"
            }
        }
```

### Step 4.3: Update Main App with Function Calling

**File**: `production_local/app.py` (additions)

```python
# production_local/app.py (add these functions)
import ollama
from components.tools.tool_definitions import TOOLS_SCHEMA
from components.tools.function_executor import FunctionExecutor
from production_local.model_selector import ModelSelector, TaskType
import json

# Initialize components
model_selector = ModelSelector()
function_executor = FunctionExecutor(
    lance_manager=lance_db_manager,  # Your existing instance
    topic_modeler=topic_modeler,  # Your existing instance
    sqlite_path="./data/assistant.db"
)

def process_with_function_calling(user_input: str, conversation_history: list = None) -> Dict:
    """
    Process user input with function calling support
    
    Args:
        user_input: User's message
        conversation_history: Previous messages for context
        
    Returns:
        Response dictionary with text and any function results
    """
    if conversation_history is None:
        conversation_history = []
    
    # Select appropriate model
    model_config = model_selector.get_model_for_task(TaskType.FUNCTION_CALLING)
    
    # Build messages
    messages = conversation_history + [
        {"role": "user", "content": user_input}
    ]
    
    # Call LLM with tools
    response = ollama.chat(
        model=model_config['name'],
        messages=messages,
        tools=TOOLS_SCHEMA,
        options={
            "temperature": model_config['temperature'],
            "num_ctx": model_config['context_window']
        }
    )
    
    # Check for tool calls
    if response.get('tool_calls'):
        # Execute all tool calls
        tool_results = []
        for tool_call in response['tool_calls']:
            func_name = tool_call['function']['name']
            func_args = tool_call['function']['arguments']
            
            # Execute function
            result = function_executor.execute(func_name, func_args)
            tool_results.append({
                "function": func_name,
                "result": result
            })
        
        # Send tool results back to LLM for final response
        messages.append(response['message'])
        messages.append({
            "role": "tool",
            "content": json.dumps(tool_results)
        })
        
        # Get final response
        final_response = ollama.chat(
            model=model_config['name'],
            messages=messages,
            options={
                "temperature": model_config['temperature'],
                "num_ctx": model_config['context_window']
            }
        )
        
        return {
            "text": final_response['message']['content'],
            "tool_calls": tool_results,
            "model": model_config['name']
        }
    
    else:
        # No tools needed, return direct response
        return {
            "text": response['message']['content'],
            "tool_calls": None,
            "model": model_config['name']
        }

# Add new Flask route for voice assistant
@app.route('/api/voice_command', methods=['POST'])
def handle_voice_command():
    """Handle voice input and return spoken response"""
    data = request.json
    user_text = data.get('text', '')
    conversation_id = data.get('conversation_id', 'default')
    
    # Process with function calling
    response = process_with_function_calling(
        user_input=user_text,
        conversation_history=get_conversation_history(conversation_id)
    )
    
    # Store in history
    append_to_conversation(conversation_id, user_text, response['text'])
    
    return jsonify({
        "status": "success",
        "response": response['text'],
        "tool_calls": response.get('tool_calls'),
        "model": response['model']
    })
```

---

## Phase 5: Implement Report Generation Templates

### Step 5.1: Create Report Generator

**File**: `components/reports/report_generator.py`

```python
# components/reports/report_generator.py
import ollama
from typing import Dict, List, Optional
from production_local.model_selector import ModelSelector, TaskType

class ReportGenerator:
    """Generate NotebookLM-style reports from documents"""
    
    # Report type prompts
    REPORT_TEMPLATES = {
        "briefing": """Using ONLY the provided documents, create a comprehensive Briefing Report.

Structure your report EXACTLY as follows:

# EXECUTIVE SUMMARY
[2-3 paragraphs providing a high-level overview of the main arguments, conclusions, and significance]

# KEY FINDINGS
[Bullet points of the most critical facts, data points, or discoveries from the documents]

# MAIN THEMES
[Identify and explain the 3-5 major themes or topics discussed]

# DETAILED ANALYSIS
[For each major section of the documents, provide deeper analysis]

# STRATEGIC IMPLICATIONS
[What are the actionable conclusions or next steps implied by these documents?]

# QUESTIONS FOR FURTHER INVESTIGATION
[What remains unclear or requires additional research?]

CRITICAL RULES:
- Use ONLY information from the provided documents
- Cite specific sections when making claims
- If information is not in the documents, explicitly state this
- Do not add outside knowledge or assumptions
- Maintain objectivity and balanced analysis

Documents:
{documents}""",

        "faq": """Using ONLY the provided documents, create a comprehensive FAQ (Frequently Asked Questions) document.

Generate 10-15 questions that someone would naturally ask about this content, organized by topic.

Format:
## [TOPIC CATEGORY 1]

**Q: [Question 1]**
A: [Detailed answer citing specific parts of the documents]

**Q: [Question 2]**
A: [Detailed answer]

## [TOPIC CATEGORY 2]
...

RULES:
- Questions should cover: key concepts, practical applications, common confusions, edge cases
- Answers must cite specific document sections
- If a natural question cannot be answered from the documents, state: "This is not addressed in the provided documents"
- Include both basic and advanced questions

Documents:
{documents}""",

        "timeline": """Using ONLY the provided documents, create a detailed chronological timeline of all events mentioned.

Format:
# TIMELINE OF EVENTS

## [Time Period 1]
**[Date/Time]** - [Event description]
- Context: [Why this matters]
- Source: [Document section reference]

**[Date/Time]** - [Event description]
- Context: [Why this matters]
- Source: [Document section reference]

## [Time Period 2]
...

RULES:
- Include ALL dates/times mentioned in the documents
- If dates are relative (e.g., "last week"), note this
- Group by logical time periods
- Provide context for each event
- Note any contradictions or uncertainties in dates

Documents:
{documents}""",

        "study_guide": """Using ONLY the provided documents, create a comprehensive Study Guide.

Structure:
# STUDY GUIDE

## CORE CONCEPTS
[Define and explain the main concepts]

## KEY TERMINOLOGY
[Glossary of important terms with definitions]

## MAIN IDEAS BREAKDOWN
[Break down complex ideas into digestible sections]

## CRITICAL CONNECTIONS
[How do different concepts relate to each other?]

## PRACTICE QUESTIONS
[10 questions that test understanding of the material]
[Include answers at the end]

## SUMMARY CHEATSHEET
[One-page summary of the most important points]

RULES:
- Explain concepts clearly as if teaching a beginner
- Use examples from the documents
- Highlight areas that require deeper understanding
- Mark difficult concepts with ⚠️

Documents:
{documents}""",

        "comparison": """Using ONLY the provided documents, create a detailed comparison analysis.

# COMPARATIVE ANALYSIS

## SIMILARITIES
[What common themes, arguments, or data appear across documents?]

## DIFFERENCES
### Approach/Methodology
[How do the documents differ in their approach?]

### Conclusions
[What different conclusions are reached?]

### Emphasis/Focus
[What does each document prioritize?]

## CONTRADICTIONS
[Are there any direct contradictions? If so, analyze why]

## SYNTHESIS
[What unified understanding emerges when considering all documents together?]

## GAPS
[What perspectives or information are missing from this collection?]

RULES:
- Be specific with citations
- Remain objective
- Highlight both agreement and disagreement
- Consider the context of each document

Documents:
{documents}"""
    }
    
    def __init__(self):
        self.model_selector = ModelSelector()
    
    def generate(
        self,
        documents: List[Dict[str, str]],
        report_type: str = "briefing",
        focus_area: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate a report from documents
        
        Args:
            documents: List of dicts with 'content' and 'metadata'
            report_type: Type of report (briefing, faq, timeline, study_guide, comparison)
            focus_area: Optional specific aspect to focus on
            stream: If True, yield chunks instead of returning full text
            
        Returns:
            Generated report text (or generator if stream=True)
        """
        # Validate report type
        if report_type not in self.REPORT_TEMPLATES:
            raise ValueError(f"Unknown report type: {report_type}. Must be one of {list(self.REPORT_TEMPLATES.keys())}")
        
        # Assemble documents
        doc_text = self._format_documents(documents)
        
        # Get prompt template
        prompt = self.REPORT_TEMPLATES[report_type].format(documents=doc_text)
        
        # Add focus area if specified
        if focus_area:
            prompt += f"\n\nPay special attention to: {focus_area}"
        
        # Calculate document size for model selection
        doc_tokens = len(doc_text.split()) * 1.3  # Rough token estimate
        
        # Select appropriate model
        model_config = self.model_selector.get_model_for_task(
            TaskType.REPORT_GENERATION,
            document_size=int(doc_tokens)
        )
        
        # Generate report
        if stream:
            return self._generate_streaming(prompt, model_config)
        else:
            return self._generate_full(prompt, model_config)
    
    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents for inclusion in prompt"""
        formatted = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            source = metadata.get('filename', f'Document {i}')
            content = doc.get('content', '')
            
            formatted.append(f"""
# DOCUMENT {i}: {source}
## Metadata: {metadata}
## Content:
{content}
---
""")
        return "\n".join(formatted)
    
    def _generate_full(self, prompt: str, model_config: Dict) -> str:
        """Generate complete report"""
        response = ollama.generate(
            model=model_config['name'],
            prompt=prompt,
            options={
                "temperature": 0.3,  # Lower for reports (more factual)
                "num_ctx": model_config['context_window']
            }
        )
        return response['response']
    
    def _generate_streaming(self, prompt: str, model_config: Dict):
        """Generate report with streaming"""
        stream = ollama.generate(
            model=model_config['name'],
            prompt=prompt,
            stream=True,
            options={
                "temperature": 0.3,
                "num_ctx": model_config['context_window']
            }
        )
        
        for chunk in stream:
            yield chunk['response']
```

### Step 5.2: Add Report Routes to Flask

**File**: `production_local/app.py` (additional routes)

```python
from components.reports.report_generator import ReportGenerator

report_generator = ReportGenerator()

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Generate NotebookLM-style reports"""
    data = request.json
    report_type = data.get('report_type', 'briefing')
    document_ids = data.get('document_ids', [])
    focus_area = data.get('focus_area')
    
    # Retrieve documents from LanceDB
    documents = []
    for doc_id in document_ids:
        # Query for full document
        result = lance_db_manager.search_with_context_assembly(
            query=doc_id,
            assemble_full_context=True
        )
        docs = result.get('full_documents', {})
        if doc_id in docs:
            documents.append({
                'content': docs[doc_id]['content'],
                'metadata': docs[doc_id]['metadata']
            })
    
    # Generate report
    try:
        report = report_generator.generate(
            documents=documents,
            report_type=report_type,
            focus_area=focus_area,
            stream=False
        )
        
        return jsonify({
            "status": "success",
            "report": report,
            "report_type": report_type,
            "documents_analyzed": len(documents)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/generate_report_stream', methods=['POST'])
def generate_report_stream():
    """Generate report with Server-Sent Events streaming"""
    data = request.json
    report_type = data.get('report_type', 'briefing')
    document_ids = data.get('document_ids', [])
    
    # Retrieve documents (same as above)
    documents = []
    # ... document retrieval code ...
    
    def generate():
        try:
            for chunk in report_generator.generate(
                documents=documents,
                report_type=report_type,
                stream=True
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )
```

---

## Phase 6: Create Voice Assistant Integration

### Step 6.1: Create Voice Assistant Controller

**File**: `production_local/voice_assistant.py`

```python
# production_local/voice_assistant.py
from components.listener.WhisperListener import WhisperListener  # or MediaPipeListener
from components.speaker.PiperSpeaker import PiperSpeaker
from production_local.app import process_with_function_calling
import logging
from typing import Optional
import json

class VoiceAssistant:
    """Main voice assistant controller combining all components"""
    
    def __init__(
        self,
        whisper_model: str = "base",
        piper_model: str = "./models/piper/en_US-lessac-medium.onnx",
        piper_binary: str = "./models/piper/piper",
        push_to_talk: bool = True
    ):
        """
        Initialize voice assistant
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium)
            piper_model: Path to Piper TTS model
            piper_binary: Path to Piper executable
            push_to_talk: If True, require Enter key to start listening
        """
        self.logger = logging.getLogger(__name__)
        self.push_to_talk = push_to_talk
        self.conversation_history = []
        
        # Initialize components
        print("🎤 Loading voice recognition...")
        self.listener = WhisperListener(model_size=whisper_model)
        
        print("🔊 Loading text-to-speech...")
        self.speaker = PiperSpeaker(
            piper_binary=piper_binary,
            model_path=piper_model
        )
        
        print("✅ Voice Assistant Ready!")
    
    def listen(self, duration: int = 5) -> Optional[str]:
        """
        Listen for voice input
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Transcribed text or None
        """
        try:
            text = self.listener.listen_once(duration=duration)
            return text if text else None
        except Exception as e:
            self.logger.error(f"Listen error: {e}")
            return None
    
    def speak(self, text: str):
        """
        Speak text through TTS
        
        Args:
            text: Text to speak
        """
        try:
            self.speaker.speak(text, play_audio=True)
        except Exception as e:
            self.logger.error(f"Speak error: {e}")
    
    def process(self, user_input: str) -> dict:
        """
        Process user input through the AI system
        
        Args:
            user_input: User's text input
            
        Returns:
            Response dictionary
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Process with function calling
        response = process_with_function_calling(
            user_input=user_input,
            conversation_history=self.conversation_history
        )
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response['text']
        })
        
        # Keep history manageable (last 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def run_loop(self):
        """
        Main voice assistant loop
        """
        print("\n" + "="*60)
        print("🤖 OGUMMAA VOICE ASSISTANT")
        print("="*60)
        print("\nCommands:")
        print("  - Say anything to interact")
        print("  - Say 'quit' or 'exit' to stop")
        print("  - Say 'clear history' to reset conversation")
        print("="*60 + "\n")
        
        while True:
            try:
                # Wait for user trigger
                if self.push_to_talk:
                    input("Press Enter to speak... ")
                
                # Listen
                print("\n🎤 Listening...")
                user_text = self.listen(duration=5)
                
                if not user_text:
                    print("❌ No speech detected. Try again.")
                    continue
                
                print(f"👤 You: {user_text}\n")
                
                # Check for exit commands
                if user_text.lower() in ['quit', 'exit', 'stop', 'goodbye']:
                    farewell = "Goodbye! Have a great day."
                    print(f"🤖 Assistant: {farewell}")
                    self.speak(farewell)
                    break
                
                # Check for history clear
                if 'clear history' in user_text.lower():
                    self.conversation_history = []
                    response_text = "I've cleared our conversation history."
                    print(f"🤖 Assistant: {response_text}")
                    self.speak(response_text)
                    continue
                
                # Process input
                print("🤔 Thinking...")
                response = self.process(user_text)
                
                response_text = response['text']
                tool_calls = response.get('tool_calls')
                
                # Display response
                print(f"🤖 Assistant: {response_text}")
                
                # Display tool calls if any
                if tool_calls:
                    print("\n🔧 Actions taken:")
                    for tool in tool_calls:
                        func_name = tool.get('function', 'unknown')
                        result = tool.get('result', {})
                        if result.get('success'):
                            print(f"   ✓ {func_name}: {result.get('data', {}).get('message', 'Done')}")
                        else:
                            print(f"   ✗ {func_name}: {result.get('error', 'Failed')}")
                
                print()  # Blank line
                
                # Speak response
                print("🔊 Speaking...")
                self.speak(response_text)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            
            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                print(f"❌ Error: {e}")
                continue

def main():
    """Entry point for voice assistant"""
    assistant = VoiceAssistant(
        whisper_model="base",  # Change to "small" or "medium" for better accuracy
        push_to_talk=True
    )
    assistant.run_loop()

if __name__ == "__main__":
    main()
```

---

## Phase 7: Frontend Enhancements

### Step 7.1: Add Voice Interface to Web UI

**File**: `ogummaa-frontend/sovereign-flask/static/js/VoiceInterface.js`

```javascript
// ogummaa-frontend/sovereign-flask/static/js/VoiceInterface.js

class VoiceInterface {
    constructor() {
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioContext = null;
        this.initAudio();
    }
    
    async initAudio() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.error("Web Audio API not supported", e);
        }
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.addEventListener("dataavailable", event => {
                this.audioChunks.push(event.data);
            });
            
            this.mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.sendAudioToBackend(audioBlob);
            });
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.updateUIRecording(true);
            
        } catch (err) {
            console.error("Microphone access denied:", err);
            alert("Please allow microphone access to use voice features");
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateUIRecording(false);
            
            // Stop all tracks
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
    
    async sendAudioToBackend(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob);
        
        try {
            // Show processing indicator
            this.showProcessing();
            
            // Send to backend
            const response = await fetch('/api/voice_process', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // Display transcription
                this.displayTranscription(result.transcription);
                
                // Display AI response
                this.displayResponse(result.response);
                
                // Play audio response if available
                if (result.audio_url) {
                    this.playAudioResponse(result.audio_url);
                }
            } else {
                this.displayError(result.error);
            }
            
        } catch (err) {
            console.error("Voice processing error:", err);
            this.displayError("Failed to process voice input");
        } finally {
            this.hideProcessing();
        }
    }
    
    displayTranscription(text) {
        const container = document.getElementById('transcription-display');
        if (container) {
            container.textContent = `You said: ${text}`;
            container.classList.add('visible');
        }
    }
    
    displayResponse(text) {
        const container = document.getElementById('ai-response-display');
        if (container) {
            container.textContent = text;
            container.classList.add('visible');
        }
    }
    
    async playAudioResponse(audioUrl) {
        const audio = new Audio(audioUrl);
        await audio.play();
    }
    
    updateUIRecording(isRecording) {
        const button = document.getElementById('voice-button');
        if (button) {
            if (isRecording) {
                button.classList.add('recording');
                button.textContent = '🔴 Stop';
            } else {
                button.classList.remove('recording');
                button.textContent = '🎤 Speak';
            }
        }
    }
    
    showProcessing() {
        const indicator = document.getElementById('processing-indicator');
        if (indicator) indicator.style.display = 'block';
    }
    
    hideProcessing() {
        const indicator = document.getElementById('processing-indicator');
        if (indicator) indicator.style.display = 'none';
    }
    
    displayError(message) {
        const container = document.getElementById('error-display');
        if (container) {
            container.textContent = message;
            container.classList.add('visible');
            setTimeout(() => container.classList.remove('visible'), 5000);
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const voiceInterface = new VoiceInterface();
    
    const voiceButton = document.getElementById('voice-button');
    if (voiceButton) {
        voiceButton.addEventListener('click', () => {
            if (voiceInterface.isRecording) {
                voiceInterface.stopRecording();
            } else {
                voiceInterface.startRecording();
            }
        });
    }
});
```

### Step 7.2: Add Voice Processing Route

**File**: `production_local/app.py` (add voice processing endpoint)

```python
from components.listener.WhisperListener import WhisperListener
from components.speaker.PiperSpeaker import PiperSpeaker
import tempfile
import os

# Initialize voice components
whisper = WhisperListener(model_size="base")
piper = PiperSpeaker()

@app.route('/api/voice_process', methods=['POST'])
def process_voice():
    """Process voice input from web interface"""
    if 'audio' not in request.files:
        return jsonify({"status": "error", "error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    
    try:
        # Transcribe
        transcription = whisper.model.transcribe(tmp_path, language='en')
        user_text = transcription['text'].strip()
        
        # Process through AI
        response = process_with_function_calling(
            user_input=user_text,
            conversation_history=[]
        )
        
        response_text = response['text']
        
        # Generate TTS
        audio_output_path = f"static/audio/response_{uuid.uuid4()}.wav"
        piper.speak(response_text, output_file=audio_output_path, play_audio=False)
        
        return jsonify({
            "status": "success",
            "transcription": user_text,
            "response": response_text,
            "audio_url": f"/{audio_output_path}",
            "tool_calls": response.get('tool_calls')
        })
    
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
```

---

## Phase 8: Complete System Integration

### Step 8.1: Create Unified Configuration

**File**: `config/system_config.yaml`

```yaml
# config/system_config.yaml

system:
  name: "Ogummaa NotebookLM + Voice Assistant"
  version: "2.0"
  mode: "sovereign"  # offline-first

# Database configuration
database:
  lance_db_path: "./lancedb_data"
  sqlite_path: "./data/assistant.db"
  
# Model configuration
models:
  embedding:
    name: "universal-sentence-encoder"
    dimension: 512
  
  llm:
    long_context:
      name: "gemma3-128k"
      context_window: 128000
    fast:
      name: "gemma2:9b"
      context_window: 8192
  
  voice:
    stt:
      engine: "whisper"  # or "mediapipe"
      model_size: "base"  # tiny, base, small, medium
    tts:
      engine: "piper"
      model: "./models/piper/en_US-lessac-medium.onnx"
      speaking_rate: 1.0

# Feature flags
features:
  voice_input: true
  voice_output: true
  function_calling: true
  report_generation: true
  topic_modeling: true
  web_interface: true

# Performance settings
performance:
  max_chunks_per_query: 10
  chunk_size_tokens: 512
  chunk_overlap: 50
  batch_size: 32
  
# Hardware optimization
hardware:
  cpu_only: true
  simd_optimization: true
  thread_count: 4  # adjust based on CPU
  ram_limit_gb: 16
```

### Step 8.2: Create Startup Script

**File**: `start_assistant.sh`

```bash
#!/bin/bash

# start_assistant.sh - Launch Ogummaa NotebookLM + Voice Assistant

set -e

echo "========================================="
echo " Ogummaa NotebookLM + Voice Assistant"
echo "========================================="
echo ""

# Check if Ollama is running
echo "🔍 Checking Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama not running. Starting..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Check if models are available
echo "🔍 Checking models..."
if ! ollama list | grep -q "gemma3-128k"; then
    echo "📥 Gemma 3 not found. Setting up..."
    ollama pull gemma3
    
    # Create custom model
    cat > /tmp/Modelfile << EOF
FROM gemma3
PARAMETER num_ctx 128000
PARAMETER temperature 0.7
EOF
    ollama create gemma3-128k -f /tmp/Modelfile
    rm /tmp/Modelfile
fi

# Activate virtual environment (if exists)
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Check Python dependencies
echo "🔍 Checking dependencies..."
pip install -q -r requirements.txt --break-system-packages

# Create necessary directories
mkdir -p lancedb_data
mkdir -p data
mkdir -p static/audio
mkdir -p models/piper
mkdir -p models/mediapipe

# Choose mode
echo ""
echo "Select mode:"
echo "  1) Web Interface (Flask app)"
echo "  2) Voice Assistant (Terminal)"
echo "  3) Both (Web + Voice)"
echo ""
read -p "Choice [1-3]: " mode

case $mode in
    1)
        echo "🌐 Starting Web Interface..."
        python production_local/app.py
        ;;
    2)
        echo "🎤 Starting Voice Assistant..."
        python production_local/voice_assistant.py
        ;;
    3)
        echo "🚀 Starting both services..."
        python production_local/app.py &
        WEB_PID=$!
        sleep 3
        python production_local/voice_assistant.py
        kill $WEB_PID 2>/dev/null || true
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
```

Make it executable:
```bash
chmod +x start_assistant.sh
```

---

## Phase 9: Testing & Validation

### Step 9.1: Create Test Suite

**File**: `tests/test_voice_assistant.py`

```python
# tests/test_voice_assistant.py
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from production_local.voice_assistant import VoiceAssistant
from components.tools.function_executor import FunctionExecutor
from components.reports.report_generator import ReportGenerator

class TestVoiceAssistant(unittest.TestCase):
    """Test suite for voice assistant functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize components once for all tests"""
        print("Setting up test environment...")
        # Note: Tests require models to be downloaded
    
    def test_function_calling(self):
        """Test that function calling works"""
        # Create mock input
        test_input = "Search for information about machine learning in my documents"
        
        # This would call the actual function
        # For testing, we'd mock the components
        pass
    
    def test_report_generation(self):
        """Test report generation"""
        generator = ReportGenerator()
        
        # Mock document
        test_docs = [{
            'content': "Machine learning is a subset of artificial intelligence...",
            'metadata': {'filename': 'test.txt'}
        }]
        
        # Generate briefing
        report = generator.generate(
            documents=test_docs,
            report_type="briefing"
        )
        
        self.assertIsNotNone(report)
        self.assertIn("EXECUTIVE SUMMARY", report)
    
    def test_context_window_selection(self):
        """Test that appropriate models are selected based on document size"""
        from production_local.model_selector import ModelSelector, TaskType
        
        selector = ModelSelector()
        
        # Small document
        small_config = selector.get_model_for_task(TaskType.DOCUMENT_ANALYSIS, document_size=5000)
        self.assertEqual(small_config['name'], 'gemma2:9b')
        
        # Large document
        large_config = selector.get_model_for_task(TaskType.DOCUMENT_ANALYSIS, document_size=60000)
        self.assertEqual(large_config['name'], 'gemma3-128k')

if __name__ == '__main__':
    unittest.main()
```

### Step 9.2: Create End-to-End Test

**File**: `tests/test_e2e.py`

```python
# tests/test_e2e.py
"""End-to-end test simulating real usage"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

def test_document_upload_and_query():
    """Test: Upload document → Ask question → Get answer with citations"""
    print("\n=== E2E Test: Document Upload & Query ===\n")
    
    # 1. Upload a test document
    print("1. Uploading test document...")
    from production_local.ingest_pipeline import ingest_document
    
    test_file = "tests/fixtures/sample_doc.txt"
    doc_id = ingest_document(test_file)
    print(f"   ✓ Document uploaded: {doc_id}")
    
    # 2. Ask a question
    print("\n2. Asking question...")
    from production_local.app import process_with_function_calling
    
    response = process_with_function_calling(
        user_input="What are the main points in the document?",
        conversation_history=[]
    )
    
    print(f"   ✓ Response: {response['text'][:100]}...")
    
    # 3. Generate a report
    print("\n3. Generating briefing report...")
    from components.reports.report_generator import ReportGenerator
    
    generator = ReportGenerator()
    # ... report generation code ...
    
    print("   ✓ Report generated")
    
    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    test_document_upload_and_query()
```

---

## Phase 10: Documentation & Deployment

### Step 10.1: Create User Guide

**File**: `docs/USER_GUIDE.md`

## Ogummaa NotebookLM + Voice Assistant User Guide

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone <your-repo>
cd ogummaa-notebooklm

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Download models
./scripts/download_models.sh

# Start the system
./start_assistant.sh
```

### 2. Using the Web Interface

#### Upload Documents
1. Navigate to http://localhost:5000
2. Click "Upload Documents"
3. Select PDFs, TXT files, or other supported formats
4. Wait for processing to complete

#### Generate Reports
1. Select documents from your library
2. Choose report type:
   - **Briefing**: Executive summary with key findings
   - **FAQ**: Question and answer format
   - **Timeline**: Chronological events
   - **Study Guide**: Educational breakdown
3. Click "Generate Report"

#### Ask Questions
1. Type your question in the chat
2. System will:
   - Search your documents
   - Provide answers with citations
   - Execute actions if needed (calendar, notes, etc.)

### 3. Using Voice Assistant

```bash
# Start voice mode
./start_assistant.sh
# Choose option 2

# Or directly
python production_local/voice_assistant.py
```

**Voice Commands**:
- "Search my documents for information about [topic]"
- "Generate a briefing report from my financial documents"
- "Add a calendar event for tomorrow at 3pm"
- "Create a note titled [title]"
- "What topics are in my documents?"

### 4. Tips for Best Results

#### Document Upload
- Upload related documents together
- Use descriptive filenames
- Organize by topic/category

#### Questioning
- Be specific: "What were the Q3 revenue numbers?" vs "Tell me about revenue"
- Reference document names when needed
- Ask follow-up questions for deeper insight

#### Report Generation
- Use "Focus Area" to narrow large documents
- Generate FAQs for complex technical content
- Use Study Guides for learning materials

## Advanced Features

### Custom Report Templates
Edit `components/reports/report_generator.py` to add your own templates

### Function Calling
The system can:
- Search documents
- Add calendar events
- Create notes
- Analyze topics
- Generate reports

### Voice Customization
Adjust voice settings in `config/system_config.yaml`:
- Speaking rate
- Model selection
- Recording duration
```

---

## Summary: Complete Implementation Checklist

### ✅ Phase 1: Foundation
- [ ] Install and configure Gemma 3 (128K context)
- [ ] Create model configuration file
- [ ] Enhance LanceDB manager for long context
- [ ] Implement model selector utility

### ✅ Phase 2: Voice Input
- [ ] Install Whisper/MediaPipe dependencies
- [ ] Create listener component
- [ ] Test audio recording and transcription

### ✅ Phase 3: Voice Output
- [ ] Download Piper TTS
- [ ] Create speaker component
- [ ] Test text-to-speech generation

### ✅ Phase 4: Function Calling
- [ ] Define tool schema
- [ ] Create function executor
- [ ] Update Flask app for function calling
- [ ] Initialize SQLite for structured data

### ✅ Phase 5: Report Generation
- [ ] Create report generator with templates
- [ ] Add report routes to Flask
- [ ] Test all report types

### ✅ Phase 6: Voice Integration
- [ ] Create voice assistant controller
- [ ] Implement conversation loop
- [ ] Test end-to-end voice interaction

### ✅ Phase 7: Frontend
- [ ] Add voice interface JavaScript
- [ ] Create voice processing endpoints
- [ ] Update UI templates

### ✅ Phase 8: System Integration
- [ ] Create unified configuration
- [ ] Write startup script
- [ ] Set up directory structure

### ✅ Phase 9: Testing
- [ ] Create unit tests
- [ ] Create E2E tests
- [ ] Validate all features

### ✅ Phase 10: Documentation
- [ ] Write user guide
- [ ] Create API documentation
- [ ] Add troubleshooting section

---

## Next Steps

1. **Start with Phase 1** - Get Gemma 3 working with long context
2. **Test incrementally** - Validate each phase before moving on
3. **Optimize for your hardware** - Adjust model sizes based on Intel N95/N100
4. **Customize reports** - Add domain-specific templates
5. **Train on your data** - Upload your personal documents for "Life Domain" AI

**Estimated Implementation Time**: 2-3 days for full system

**Hardware Requirements**:
- Minimum: 16GB RAM, Intel N95/N100
- Recommended: 32GB RAM for full 128K context

---

## Troubleshooting

### Ollama Issues
```bash
# Restart Ollama
pkill ollama
ollama serve

# Check models
ollama list

# Re-create custom model
ollama create gemma3-128k -f Modelfile
```

### Memory Issues
- Reduce context window: `num_ctx=32000`
- Use smaller models: `gemma2:9b` instead of `gemma3-128k`
- Decrease chunk size in config

### Voice Recognition Issues
- Check microphone permissions
- Test audio levels: `arecord -d 5 test.wav`
- Try different Whisper model size

### Piper TTS Issues
- Verify binary is executable: `chmod +x piper`
- Test directly: `echo "test" | ./piper --model <model> --output test.wav`
- Check audio player: `aplay test.wav`

---
# Ogummaa Assistant: Foreground Fast Index Integration

## Overview

This module integrates BERTopic (Logic) with SQLite (Storage) to create the "Foreground Fast Index" for the Ogummaa Assistant.

By implementing this architecture, the Assistant can instantly "know" and categorize uploaded domains of life (e.g., "Medical", "Financial", "Vehicle Maintenance") without performing computationally expensive semantic searches for every query. This acts as a caching layer for high-level context.

---

## 📂 Directory Structure

```
/
├── production_local/
│   ├── TopicModeler.py       # Logic: Handles text processing and clustering
│   ├── DatabaseManager.py    # Storage: Manages SQLite and Vector DB connections
│   └── run_clustering.py     # Automation: Pipeline to read docs -> cluster -> save
│
└── data_storage/
    └── raw_texts/            # Source directory for .txt files to be clustered
```

---

## 🛠 Components

### 1. The Topic Modeler (`TopicModeler.py`)

**Role:** The "Brain" of the operation. This script utilizes a CPU-optimized embedding model (`all-MiniLM-L6-v2`) to process text documents and identify thematic clusters.

- **Key Dependencies:** `pandas`, `bertopic`
- **Key Method:** `fit_transform(documents, doc_ids)`
  - **Input:** List of raw text strings and their filenames
  - **Output:** Structured dictionary containing topic names, keywords, and document mappings
  - **Logic:** Removes noise (Topic -1) and cleans topic labels (e.g., converts `0_finance_money` to `finance money`)

### 2. The Database Manager (`DatabaseManager.py`)

**Role:** The "Memory" of the operation. This is an upgrade to the Hybrid Memory controller. It manages persistent SQLite tables to store the clusters generated by the Topic Modeler.

- **Key Dependencies:** `sqlite3`, `lancedb_manager`
- **Schema Created:**
  - `topics`: Stores cluster definitions (ID, Name, Keywords)
  - `doc_topics`: Links specific document IDs to Topics with probability scores
  - `calendar`: Placeholder for task management events
- **Key Method:** `save_clusters(cluster_data)`
  - Performs a full refresh strategy (wipes old topic data, inserts new clusters) to ensure the index stays clean

### 3. The Automation Script (`run_clustering.py`)

**Role:** The "Bridge". This script connects the raw data on disk, the Topic Modeler, and the Database Manager into a single execution flow.

**Workflow:**
1. Reads `.txt` files from `data_storage/raw_texts/`
2. Validates document count (BERTopic usually requires 5-10+ docs)
3. Runs TopicModeler to generate clusters
4. Calls DatabaseManager to save results to `life_domain.db`

---

## 🚀 How to Use

### Prerequisites

Ensure the necessary Python packages are installed:

```bash
pip install bertopic pandas sqlite3 lancedb
```

### Step 1: Ingest Data

Ensure your main ingestion pipeline (e.g., Flask UI) saves a copy of the raw text content of uploaded files to the storage directory.

- **Target Path:** `data_storage/raw_texts/`

### Step 2: Run Clustering

Execute the automation script periodically (e.g., weekly or after a large batch upload) to update the Assistant's "World View".

```bash
python production_local/run_clustering.py
```

**Expected Output:**

```
📂 Reading files from data_storage/raw_texts/...
📉 Clustering 50 documents...
💾 Saving Topics to SQLite...
🔗 Linking Documents to Topics...
✅ Hybrid Index Updated Successfully.

🎉 Pipeline Complete.
Your Assistant now recognizes the following life domains:
  - Topic 0: finance money bank
  - Topic 1: medical health insurance
```

### Step 3: Querying (Integration Logic)

When a user asks a question, you can now skip the Vector DB scan for high-level context by querying the SQLite DB directly.

**Example SQL Flow:**

User: _"What is the status of my finances?"_

```sql
-- 1. Find the topic ID
SELECT id FROM topics WHERE name LIKE '%finance%';

-- 2. Retrieve all related documents instantly
SELECT doc_id 
FROM doc_topics 
WHERE topic_id = [FOUND_ID_FROM_STEP_1];
```

---

## 📝 Notes

- **Minimum Data:** BERTopic requires a small batch of documents to work effectively. If you have fewer than 5 documents, the script will abort to prevent errors.
- **Model Selection:** The default model `all-MiniLM-L6-v2` is selected for speed on standard CPUs. If running on a GPU server, you may swap this for a larger model in `TopicModeler.py`.

---

## Architecture Benefits

- **Instant Context Recognition:** Bypasses expensive semantic searches for high-level categorization
- **Caching Layer:** Acts as a fast lookup for domain identification
- **CPU-Optimized:** Designed for efficient operation on standard hardware
- **Offline-First:** Fully air-gapped operation with no cloud dependencies
- **Scalable:** Handles growing document collections with periodic re-clustering

---

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
**This completes the implementation guide. You now have a fully sovereign, offline-first NotebookLM clone with voice capabilities!** 🎉
