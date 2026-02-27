#!/usr/bin/env python3
"""
Local LLM Benchmark Tool
========================
Compare local LLM performance across MLX and Ollama backends.
Includes structured extraction evaluation with F1 scoring.

Usage:
    python benchmark.py                      # Interactive mode
    python benchmark.py --model 1            # Run model 1 (non-interactive)
    python benchmark.py --all --skip-warnings  # Run all models
"""

import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from jinja2 import Environment, FileSystemLoader


# =============================================================================
# Constants
# =============================================================================

# Memory thresholds (GB)
MEMORY_COMFORTABLE_HEADROOM_GB = 4  # Extra GB for "fits" indicator

# MLX server timing (seconds)
MLX_STARTUP_TIMEOUT_SEC = 90
MLX_STARTUP_PROGRESS_INTERVAL_SEC = 15
MLX_STARTUP_SLEEP_SEC = 3
MLX_SHUTDOWN_SLEEP_SEC = 2
PROCESS_WAIT_TIMEOUT_SEC = 5

# Ollama timing (seconds)
OLLAMA_STOP_SLEEP_SEC = 2

# API defaults
DEFAULT_TEMPERATURE = 0.3  # Lower for structured extraction
DEFAULT_MAX_TOKENS = 8192  # Enough for thinking + output if needed

# CSV logging
CSV_FILENAME = "benchmark_log.csv"
CSV_COLUMNS = [
    "timestamp", "model_name", "model_id", "backend",
    "prompt_hash", "prompt_preview",
    "temperature", "max_tokens",
    "time_sec", "tokens", "prompt_tokens", "completion_tokens",
    "memory_delta_gb",
    "f1", "precision", "recall", "json_valid",
    "success", "error"
]
PROMPT_PREVIEW_LENGTH = 60

# Prompt limits
MAX_PROMPT_SIZE_BYTES = 1_000_000  # 1MB limit

# Evaluation
FUZZY_MATCH_THRESHOLD = 0.8


# =============================================================================
# Validation Functions
# =============================================================================

def validate_model_config(model: dict, index: int) -> None:
    """Validate a model config has required fields.

    Args:
        model: Model configuration dictionary
        index: Zero-based index for error messages

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ["model_id", "backend"]
    for field in required_fields:
        if field not in model:
            raise ValueError(f"Model {index + 1} missing required field: {field}")

    if model["backend"] not in ("mlx", "ollama"):
        raise ValueError(f"Model {index + 1} has invalid backend: {model['backend']}")


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_f1_metrics(tp: int, fp: int, fn: int) -> dict:
    """Calculate precision, recall, and F1 score from counts.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Dict with 'precision', 'recall', and 'f1' keys
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def get_prompt_hash(prompt: str) -> str:
    """Get first 8 characters of SHA256 hash of prompt."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def get_display_name(model: dict) -> str:
    """Derive display name from backend and model_id.

    Args:
        model: Model configuration dictionary with 'backend' and 'model_id'

    Returns:
        Display name in 'backend/model_id' format
    """
    return f"{model['backend']}/{model['model_id']}"


def save_prompt_if_new(results_dir: Path, prompt: str) -> str:
    """Save prompt to file if not already saved. Returns hash.

    Raises:
        ValueError: If prompt exceeds MAX_PROMPT_SIZE_BYTES
    """
    prompt_bytes = len(prompt.encode('utf-8'))
    if prompt_bytes > MAX_PROMPT_SIZE_BYTES:
        raise ValueError(
            f"Prompt size ({prompt_bytes} bytes) exceeds maximum "
            f"of {MAX_PROMPT_SIZE_BYTES} bytes"
        )

    prompt_hash = get_prompt_hash(prompt)
    prompts_dir = results_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = prompts_dir / f"{prompt_hash}.txt"
    if not prompt_file.exists():
        with open(prompt_file, "w") as f:
            f.write(prompt)

    return prompt_hash


def append_to_csv(results_dir: Path, row: dict):
    """Append a row to the benchmark log CSV."""
    csv_file = results_dir / CSV_FILENAME
    file_exists = csv_file.exists()

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_csv_results(results_dir: Path) -> list[dict]:
    """Load all results from the benchmark log CSV."""
    csv_file = results_dir / CSV_FILENAME
    if not csv_file.exists():
        return []

    results = []
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for field in ["time_sec", "tokens", "prompt_tokens", "completion_tokens",
                          "memory_delta_gb", "f1", "precision", "recall"]:
                if row.get(field):
                    try:
                        row[field] = float(row[field])
                    except ValueError:
                        pass
            for field in ["temperature", "max_tokens"]:
                if row.get(field):
                    try:
                        row[field] = float(row[field])
                    except ValueError:
                        pass
            row["success"] = row.get("success", "").lower() == "true"
            row["json_valid"] = row.get("json_valid", "").lower() == "true"
            results.append(row)
    return results


def get_last_run_time(results_dir: Path, model_name: str) -> Optional[str]:
    """Get the last run timestamp for a model from CSV."""
    results = load_csv_results(results_dir)
    for row in reversed(results):
        if row.get("model_name") == model_name:
            return row.get("timestamp", "")[:16]  # Just date and time
    return None


def get_ollama_model_size_gb(model_id: str) -> Optional[float]:
    """Get Ollama model size in GB from 'ollama list'."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3 and parts[0] == model_id:
                size_str = parts[2]  # e.g., "5.2" or "18"
                unit = parts[3] if len(parts) > 3 else "GB"
                size = float(size_str)
                if "MB" in unit:
                    size /= 1024
                return size
        return None
    except Exception:
        return None


def get_mlx_model_size_gb(model_id: str) -> Optional[float]:
    """Get MLX model size in GB from HuggingFace cache.

    Only counts .safetensors files (actual model weights) to avoid
    overcounting from multiple snapshots or metadata files.
    """
    try:
        # Convert model_id to cache directory name
        # e.g., "mlx-community/Qwen3-8B-4bit" -> "models--mlx-community--Qwen3-8B-4bit"
        cache_name = "models--" + model_id.replace("/", "--")
        cache_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_name

        if not cache_path.exists():
            return None

        # Only count .safetensors files (model weights)
        # Use resolve() to follow symlinks and get actual file size
        seen_files = set()
        total_size = 0
        for f in cache_path.rglob("*.safetensors"):
            # Resolve symlinks to avoid double-counting
            real_path = f.resolve()
            if real_path not in seen_files:
                seen_files.add(real_path)
                total_size += real_path.stat().st_size

        return total_size / (1024 ** 3) if total_size > 0 else None
    except Exception:
        return None


def estimate_model_memory_gb(model_config: dict) -> float:
    """Estimate memory needed for a model.

    If estimated_memory_gb is provided in config, use it.
    Otherwise, detect model size and estimate with modest overhead.
    """
    # Use config value if provided
    if "estimated_memory_gb" in model_config:
        return model_config["estimated_memory_gb"]

    model_id = model_config.get("model_id", "")
    backend = model_config.get("backend", "")

    # Try to detect model size
    model_size = None
    if backend == "ollama":
        model_size = get_ollama_model_size_gb(model_id)
    elif backend == "mlx":
        model_size = get_mlx_model_size_gb(model_id)

    if model_size:
        # Estimate: model size + 20% overhead (KV cache, runtime) + 1GB buffer
        return round(model_size * 1.2 + 1, 1)

    # Default fallback
    return 8.0


# =============================================================================
# System Memory Functions
# =============================================================================

def get_system_info() -> dict:
    """Get detailed system info for reports."""
    info = {
        "os_version": "-",
        "cpu": "-",
        "ram_gb": 0,
        "gpu": "-",
    }

    if platform.system() == "Darwin":
        # macOS version
        try:
            result = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
            )
            info["os_version"] = f"macOS {result.stdout.strip()}"
        except Exception:
            pass

        # CPU info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            cpu = result.stdout.strip()
            if not cpu:
                # Apple Silicon - get chip name
                result = subprocess.run(
                    ["sysctl", "-n", "hw.model"],
                    capture_output=True,
                    text=True,
                )
                model = result.stdout.strip()
                # Try to get chip type from system_profiler
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.split("\n"):
                    if "Chip:" in line:
                        cpu = line.split(":")[1].strip()
                        break
                if not cpu:
                    cpu = model
            info["cpu"] = cpu
        except Exception:
            pass

        # GPU / Neural Engine info (for Apple Silicon, unified memory)
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.split("\n"):
                if "Chipset Model:" in line:
                    info["gpu"] = line.split(":")[1].strip()
                    break
            # For Apple Silicon, GPU uses unified memory
            if "Apple" in info.get("cpu", ""):
                info["gpu"] = f"{info['cpu']} GPU (unified memory)"
        except Exception:
            pass

    # RAM
    info["ram_gb"] = round(get_system_memory_gb())

    return info


def get_system_memory_gb() -> float:
    """Get total system RAM in GB.

    Returns 8.0 as a safe default for unsupported platforms.
    """
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip()) / (1024**3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    # Fallback for unsupported platforms or errors
    return 8.0


def get_available_memory_gb() -> float:
    """Get available (free + inactive) RAM in GB."""
    if platform.system() == "Darwin":
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        output = result.stdout

        page_size = 16384
        pages_free = 0
        pages_inactive = 0
        pages_purgeable = 0

        for line in output.split("\n"):
            if "page size of" in line:
                match = re.search(r"(\d+) bytes", line)
                if match:
                    page_size = int(match.group(1))
            elif "Pages free:" in line:
                pages_free = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages inactive:" in line:
                pages_inactive = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages purgeable:" in line:
                pages_purgeable = int(line.split(":")[1].strip().rstrip("."))

        available_bytes = (pages_free + pages_inactive + pages_purgeable) * page_size
        return available_bytes / (1024**3)
    else:
        result = subprocess.run(["free", "-b"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                return int(parts[6]) / (1024**3)
    return 0


def get_memory_pressure() -> dict:
    """Get detailed memory stats for logging."""
    stats = {
        "total_gb": get_system_memory_gb(),
        "available_gb": get_available_memory_gb(),
        "timestamp": datetime.now().isoformat(),
    }
    stats["used_gb"] = stats["total_gb"] - stats["available_gb"]
    # Guard against division by zero
    stats["used_percent"] = (
        (stats["used_gb"] / stats["total_gb"]) * 100
        if stats["total_gb"] > 0
        else 0
    )
    return stats


def get_memory_fit_indicator(estimated_gb: float, available_gb: float, buffer_gb: float) -> str:
    """Return a memory fit indicator string."""
    required = estimated_gb + buffer_gb
    if available_gb >= required + MEMORY_COMFORTABLE_HEADROOM_GB:
        return "✓ fits"
    elif available_gb >= required:
        return "⚠ tight"
    else:
        return "✗ won't fit"


# =============================================================================
# MLX Server Management
# =============================================================================

def is_mlx_server_running(port: int = 8080) -> bool:
    """Check if MLX server is running on the specified port."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def verify_mlx_server_health(port: int = 8080) -> bool:
    """Verify MLX server is responding to API requests.

    This provides a stronger check than just port availability,
    ensuring the server is actually ready to handle requests.
    """
    try:
        req = urllib.request.Request(f"http://localhost:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def kill_mlx_processes() -> None:
    """Kill any running MLX server processes."""
    subprocess.run(["pkill", "-f", "mlx_lm.server"], capture_output=True)
    subprocess.run(["pkill", "-f", "mlx_lm server"], capture_output=True)


def start_mlx_server(model_id: str, port: int = 8080) -> Optional[subprocess.Popen]:
    """Start MLX server for a specific model.

    Args:
        model_id: The model identifier
        port: Port to run the server on

    Returns:
        The subprocess.Popen object if successful, None otherwise
    """
    print(f"  Starting MLX server for {model_id}...")
    print(f"  This may take 30-60 seconds to load the model...")

    kill_mlx_processes()
    time.sleep(MLX_STARTUP_SLEEP_SEC)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            model_id,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for i in range(MLX_STARTUP_TIMEOUT_SEC):
        # Check if port is open first (faster check)
        if is_mlx_server_running(port):
            # Then verify the server is actually responding to API requests
            if verify_mlx_server_health(port):
                print(f"  ✓ MLX server ready on port {port} (took {i+1}s)")
                return proc

        if proc.poll() is not None:
            print(f"  ✗ MLX server process died (exit code: {proc.returncode})")
            return None

        if i % MLX_STARTUP_PROGRESS_INTERVAL_SEC == MLX_STARTUP_PROGRESS_INTERVAL_SEC - 1:
            print(f"  Still loading model... ({i+1}s)")
        time.sleep(1)

    print(f"  ✗ MLX server failed to start within {MLX_STARTUP_TIMEOUT_SEC} seconds")
    proc.terminate()
    return None


def stop_mlx_server(proc: Optional[subprocess.Popen] = None) -> None:
    """Stop MLX server."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=PROCESS_WAIT_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()

    kill_mlx_processes()
    time.sleep(MLX_SHUTDOWN_SLEEP_SEC)


# =============================================================================
# Process Checks (Pre/Post Benchmark)
# =============================================================================

def check_mlx_processes() -> bool:
    """Check if any MLX server processes are running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "mlx_lm.server"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_ollama_models_loaded() -> bool:
    """Check if any Ollama models are loaded in memory.

    Uses 'ollama ps' to check for loaded models. The Ollama service
    running in the menu bar with no models loaded is fine and won't
    trigger a warning.
    """
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        # ollama ps returns a header line + one line per loaded model
        # If only header (or empty), no models are loaded
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        # More than 1 line means models are loaded (first line is header)
        return len(lines) > 1
    except Exception:
        return False


def check_llm_processes() -> dict:
    """Check if MLX server is running or Ollama has models loaded.

    Returns dict with 'mlx' and 'ollama' boolean status.
    """
    return {
        "mlx": check_mlx_processes(),
        "ollama": check_ollama_models_loaded(),
    }


def print_process_warning(processes: dict, stage: str) -> bool:
    """Print warning if LLM processes are running or models loaded.

    Args:
        processes: dict from check_llm_processes()
        stage: 'pre' or 'post' for messaging context

    Returns:
        True if any warnings were printed, False otherwise.
    """
    running = []
    if processes["mlx"]:
        running.append("MLX server")
    if processes["ollama"]:
        running.append("Ollama model(s)")

    if not running:
        return False

    processes_str = " and ".join(running)

    if stage == "pre":
        print(f"\n⚠ WARNING: {processes_str} already loaded in memory!")
        print("  This may affect benchmark results or cause conflicts.")
        if processes["mlx"]:
            print("  To stop MLX:    pkill -f mlx_lm.server")
        if processes["ollama"]:
            print("  To unload Ollama: ollama stop <model_name>")
    else:  # post
        print(f"\n⚠ WARNING: {processes_str} still loaded in memory!")
        print("  You may want to unload these to free memory.")
        if processes["mlx"]:
            print("  To stop MLX:    pkill -f mlx_lm.server")
        if processes["ollama"]:
            print("  To unload Ollama: ollama stop <model_name>")

    return True


# =============================================================================
# Ollama Management
# =============================================================================

def is_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def stop_ollama_model(model_id: str) -> None:
    """Unload an Ollama model from memory."""
    print(f"  Unloading Ollama model: {model_id}...")
    result = subprocess.run(
        ["ollama", "stop", model_id],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("  ✓ Model unloaded")
    # Model might not be loaded, which is fine
    time.sleep(OLLAMA_STOP_SLEEP_SEC)


# =============================================================================
# Evaluation Functions
# =============================================================================

def normalize_string(s: str) -> str:
    """Normalize string for comparison."""
    if s is None:
        return ""
    return re.sub(r'\s+', ' ', str(s).lower().strip())


def fuzzy_match(pred: str, truth: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """Check if predicted string fuzzy matches truth."""
    pred_norm = normalize_string(pred)
    truth_norm = normalize_string(truth)

    if not pred_norm or not truth_norm:
        return pred_norm == truth_norm

    # Exact match
    if pred_norm == truth_norm:
        return True

    # Substring match (either direction)
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return True

    # Token overlap (Jaccard-like)
    pred_tokens = set(pred_norm.split())
    truth_tokens = set(truth_norm.split())

    if not pred_tokens or not truth_tokens:
        return False

    intersection = len(pred_tokens & truth_tokens)
    union = len(pred_tokens | truth_tokens)

    return (intersection / union) >= threshold


def try_parse_json(json_str: str) -> Optional[dict]:
    """Try to parse JSON, with basic repair for common LLM errors."""
    # First try direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try fixing extra trailing braces (common LLM error)
    # Count opening and closing braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if close_braces > open_braces:
        # Remove extra closing braces from the end
        fixed = json_str.rstrip()
        for _ in range(close_braces - open_braces):
            if fixed.endswith('}'):
                fixed = fixed[:-1].rstrip()
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Try fixing missing closing braces
    if open_braces > close_braces:
        fixed = json_str + ('}' * (open_braces - close_braces))
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return None


def extract_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON from model response, handling various formats."""
    if not response:
        return None

    # Remove thinking blocks (Qwen style)
    response = re.sub(r'<think>[\s\S]*?</think>', '', response)

    # Remove special channel/message tags (GPT-OSS style: <|channel|>, <|message|>, etc.)
    # First, try to extract just the "final" channel content if present
    final_match = re.search(r'<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\||$)', response)
    if final_match:
        response = final_match.group(1)
    else:
        # Remove all <|...|> style tags
        response = re.sub(r'<\|[^|]*\|>', '', response)

    # Remove any other XML-like tags
    response = re.sub(r'<[^>]+>', '', response)

    # Try to find JSON in code blocks first
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, response)
        if match:
            result = try_parse_json(match.group(1))
            if result:
                return result

    # Try to find raw JSON (object starting with {)
    # First try greedy match
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        result = try_parse_json(json_match.group())
        if result:
            return result

    # If greedy failed, try to find balanced JSON by scanning for complete objects
    # This handles cases where valid JSON is followed by garbage text
    start_idx = response.find('{')
    if start_idx != -1:
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(response[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found a complete JSON object
                    candidate = response[start_idx:i+1]
                    result = try_parse_json(candidate)
                    if result:
                        return result
                    break

    # Try parsing the whole stripped response
    return try_parse_json(response.strip())


def evaluate_extraction(predicted: dict, ground_truth: dict) -> dict:
    """
    Evaluate structured extraction using F1 scores.

    Returns dict with:
    - json_valid: bool
    - scores: dict of F1 scores per field type
    - overall_f1: weighted average F1
    - details: breakdown of matches
    """
    result = {
        "json_valid": predicted is not None,
        "scores": {},
        "overall_f1": 0.0,
        "details": {},
    }

    if not predicted:
        return result

    all_tp, all_fp, all_fn = 0, 0, 0

    # Evaluate attendees (simple list matching)
    if "attendees" in ground_truth:
        truth_attendees = set(normalize_string(a) for a in ground_truth.get("attendees", []))
        pred_attendees = set(normalize_string(a) for a in predicted.get("attendees", []))

        tp = len(truth_attendees & pred_attendees)
        fp = len(pred_attendees - truth_attendees)
        fn = len(truth_attendees - pred_attendees)

        result["scores"]["attendees"] = calculate_f1_metrics(tp, fp, fn)
        result["details"]["attendees"] = {"tp": tp, "fp": fp, "fn": fn}
        all_tp += tp; all_fp += fp; all_fn += fn

    # Evaluate meeting_date
    if "meeting_date" in ground_truth:
        truth_date = normalize_string(ground_truth.get("meeting_date", ""))
        pred_date = normalize_string(predicted.get("meeting_date", ""))

        match = fuzzy_match(pred_date, truth_date)
        tp, fp, fn = (1, 0, 0) if match else (0, 1 if pred_date else 0, 1)

        result["scores"]["meeting_date"] = {"precision": tp, "recall": tp, "f1": tp}
        result["details"]["meeting_date"] = {"match": match, "predicted": pred_date, "truth": truth_date}
        all_tp += tp; all_fp += fp; all_fn += fn

    # Evaluate projects (complex nested objects)
    if "projects" in ground_truth:
        truth_projects = ground_truth.get("projects", [])
        pred_projects = predicted.get("projects", [])

        project_tp, project_fp, project_fn = 0, 0, 0
        project_details = []

        # Match projects by name (more flexible matching)
        truth_by_name = {normalize_string(p.get("name", "")): p for p in truth_projects}
        pred_by_name = {normalize_string(p.get("name", "")): p for p in pred_projects}

        matched_truth = set()
        for pred_name, pred_proj in pred_by_name.items():
            # Find best matching truth project
            # Try fuzzy match first, then check if truth name is contained in pred name
            best_match = None
            for truth_name in truth_by_name:
                if truth_name in matched_truth:
                    continue
                # Check fuzzy match or if one contains the other
                if (fuzzy_match(pred_name, truth_name) or
                    truth_name in pred_name or
                    pred_name in truth_name or
                    # Also check if key words overlap (e.g., "churn" matches "customer churn prediction")
                    any(word in pred_name for word in truth_name.split() if len(word) > 3)):
                    best_match = truth_name
                    break

            if best_match and best_match not in matched_truth:
                matched_truth.add(best_match)
                truth_proj = truth_by_name[best_match]

                # Count field matches within project
                fields = ["hypothesis", "decision", "owner", "deadline"]
                field_matches = 0
                for field in fields:
                    if fuzzy_match(str(pred_proj.get(field, "")), str(truth_proj.get(field, ""))):
                        field_matches += 1

                # Methods (list)
                truth_methods = set(normalize_string(m) for m in truth_proj.get("methods", []))
                pred_methods = set(normalize_string(m) for m in pred_proj.get("methods", []))
                method_matches = len(truth_methods & pred_methods)

                project_tp += 1 + field_matches + method_matches
                project_fn += (len(fields) - field_matches) + (len(truth_methods) - method_matches)

                project_details.append({
                    "name": pred_name,
                    "matched": True,
                    "field_matches": field_matches,
                    "method_matches": method_matches,
                })
            else:
                project_fp += 1
                project_details.append({"name": pred_name, "matched": False})

        # Count unmatched truth projects
        for truth_name in truth_by_name:
            if truth_name not in matched_truth:
                truth_proj = truth_by_name[truth_name]
                project_fn += 1 + 4 + len(truth_proj.get("methods", []))  # name + fields + methods

        result["scores"]["projects"] = calculate_f1_metrics(project_tp, project_fp, project_fn)
        result["details"]["projects"] = project_details
        all_tp += project_tp; all_fp += project_fp; all_fn += project_fn

    # Evaluate action_items
    if "action_items" in ground_truth:
        truth_items = ground_truth.get("action_items", [])
        pred_items = predicted.get("action_items", [])

        item_tp, item_fp, item_fn = 0, 0, 0

        matched_truth = set()
        for pred_item in pred_items:
            pred_task = normalize_string(pred_item.get("task", ""))

            best_match_idx = None
            for i, truth_item in enumerate(truth_items):
                if i in matched_truth:
                    continue
                truth_task = normalize_string(truth_item.get("task", ""))
                if fuzzy_match(pred_task, truth_task):
                    best_match_idx = i
                    break

            if best_match_idx is not None:
                matched_truth.add(best_match_idx)
                truth_item = truth_items[best_match_idx]

                # Check assignee and due_date
                assignee_match = fuzzy_match(
                    str(pred_item.get("assignee", "")),
                    str(truth_item.get("assignee", ""))
                )
                date_match = fuzzy_match(
                    str(pred_item.get("due_date", "")),
                    str(truth_item.get("due_date", ""))
                )

                item_tp += 1 + int(assignee_match) + int(date_match)
                item_fn += (1 - int(assignee_match)) + (1 - int(date_match))
            else:
                item_fp += 1

        # Unmatched truth items
        item_fn += (len(truth_items) - len(matched_truth)) * 3  # task + assignee + due_date

        result["scores"]["action_items"] = calculate_f1_metrics(item_tp, item_fp, item_fn)
        result["details"]["action_items"] = {"matched": len(matched_truth), "total_truth": len(truth_items), "total_pred": len(pred_items)}
        all_tp += item_tp; all_fp += item_fp; all_fn += item_fn

    # Calculate overall micro-F1
    overall_metrics = calculate_f1_metrics(all_tp, all_fp, all_fn)
    result["overall_f1"] = overall_metrics["f1"]
    result["overall_precision"] = overall_metrics["precision"]
    result["overall_recall"] = overall_metrics["recall"]

    return result


# =============================================================================
# Benchmark Execution
# =============================================================================

def run_benchmark(
    backend: str,
    model_id: str,
    prompt: str,
    timeout: int = 600,
    mlx_port: int = 8080,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Run benchmark by calling the model API directly."""
    result = {
        "success": False,
        "response": "",
        "time_seconds": 0,
        "error": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if backend == "mlx":
        api_url = f"http://localhost:{mlx_port}/v1/chat/completions"
    elif backend == "ollama":
        api_url = "http://localhost:11434/v1/chat/completions"
    else:
        result["error"] = f"Unknown backend: {backend}"
        return result

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print(f"  Calling {backend.upper()} API: {api_url}")
    print(f"  Model: {model_id}")
    print(f"  Generating", end="", flush=True)

    start_time = time.time()

    # Start a thread to print elapsed time
    import threading
    stop_progress = threading.Event()

    def print_progress():
        while not stop_progress.is_set():
            elapsed = time.time() - start_time
            print(f"\r  Generating... {elapsed:.0f}s", end="", flush=True)
            stop_progress.wait(2)

    progress_thread = threading.Thread(target=print_progress, daemon=True)
    progress_thread.start()

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            stop_progress.set()
            result["time_seconds"] = time.time() - start_time
            response_data = json.loads(response.read().decode("utf-8"))

            if "choices" in response_data and len(response_data["choices"]) > 0:
                result["response"] = response_data["choices"][0]["message"]["content"]
                result["success"] = True

            if "usage" in response_data:
                result["prompt_tokens"] = response_data["usage"].get("prompt_tokens", 0)
                result["completion_tokens"] = response_data["usage"].get("completion_tokens", 0)
                result["total_tokens"] = response_data["usage"].get("total_tokens", 0)

    except urllib.error.HTTPError as e:
        stop_progress.set()
        result["time_seconds"] = time.time() - start_time
        error_body = e.read().decode("utf-8") if e.fp else ""
        result["error"] = f"HTTP {e.code}: {e.reason}\n{error_body[:500]}"

    except urllib.error.URLError as e:
        stop_progress.set()
        result["time_seconds"] = time.time() - start_time
        result["error"] = f"Connection error: {e.reason}"

    except TimeoutError:
        stop_progress.set()
        result["time_seconds"] = timeout
        result["error"] = f"Timeout after {timeout} seconds"

    except Exception as e:
        stop_progress.set()
        result["time_seconds"] = time.time() - start_time
        result["error"] = f"{type(e).__name__}: {str(e)}"

    print()  # Newline after progress
    return result


# =============================================================================
# Results Management
# =============================================================================

def save_response(
    results_dir: Path,
    model_config: dict,
    benchmark_result: dict,
    timestamp: str,
):
    """Save raw response to responses folder."""
    responses_dir = results_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp and model name
    safe_name = sanitize_filename(get_display_name(model_config))
    timestamp_safe = timestamp.replace(":", "-")
    response_file = responses_dir / f"{safe_name}_{timestamp_safe}.txt"

    with open(response_file, "w") as f:
        f.write(benchmark_result.get("response", "No response captured"))

    return response_file


def sanitize_filename(name: str) -> str:
    """Convert model name to safe directory name."""
    return re.sub(r"[^\w\-]", "-", name.lower()).strip("-")


def generate_report(results_dir: Path):
    """Generate markdown summary report from CSV data."""
    results = load_csv_results(results_dir)
    if not results:
        return None

    report_file = results_dir / "report.md"

    # Sort by timestamp (most recent last)
    results = sorted(results, key=lambda r: r.get("timestamp", ""))

    has_evaluation = any(r.get("f1") for r in results)

    with open(report_file, "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**System RAM:** {get_system_memory_gb():.0f} GB\n")
        f.write(f"**Total runs:** {len(results)}\n\n")

        # Results table with timestamp
        f.write("## Results\n\n")
        if has_evaluation:
            f.write("| Timestamp | Model | Backend | Memory Δ | Time | Tokens | F1 | Precision | Recall |\n")
            f.write("|-----------|-------|---------|----------|------|--------|-----|-----------|--------|\n")
        else:
            f.write("| Timestamp | Model | Backend | Memory Δ | Time | Tokens |\n")
            f.write("|-----------|-------|---------|----------|------|--------|\n")

        for r in results:
            timestamp = r.get("timestamp", "")[:16]  # Just date and time
            model_name = r.get("model_name", "-")
            backend = r.get("backend", "-")
            success = r.get("success", False)

            time_str = f"{r.get('time_sec', 0):.1f}s" if success else "failed"
            tokens_str = str(int(r.get("tokens", 0))) if success and r.get("tokens") else "-"
            mem_delta = r.get("memory_delta_gb", 0)
            mem_str = f"{mem_delta:.1f} GB" if mem_delta else "-"

            if has_evaluation:
                f1 = r.get("f1", 0) or 0
                precision = r.get("precision", 0) or 0
                recall = r.get("recall", 0) or 0
                f.write(f"| {timestamp} | {model_name} | {backend} | {mem_str} | {time_str} | {tokens_str} | {f1:.1%} | {precision:.1%} | {recall:.1%} |\n")
            else:
                f.write(f"| {timestamp} | {model_name} | {backend} | {mem_str} | {time_str} | {tokens_str} |\n")

        # Prompt versions
        prompts_dir = results_dir / "prompts"
        if prompts_dir.exists():
            prompt_files = list(prompts_dir.glob("*.txt"))
            if prompt_files:
                f.write("\n## Prompts Used\n\n")
                for pf in sorted(prompt_files):
                    f.write(f"- `{pf.stem}` - [view](./prompts/{pf.name})\n")

        # Recent responses
        responses_dir = results_dir / "responses"
        if responses_dir.exists():
            response_files = sorted(responses_dir.glob("*.txt"), reverse=True)[:10]
            if response_files:
                f.write("\n## Recent Responses\n\n")
                for rf in response_files:
                    f.write(f"- [{rf.stem}](./responses/{rf.name})\n")

        f.write("\n---\n")
        f.write("*Generated by local-llm-benchmark*\n")

    return report_file


def generate_html_report(results_dir: Path):
    """Generate interactive HTML report with sortable table and charts."""
    results = load_csv_results(results_dir)
    if not results:
        return None

    report_file = results_dir / "report.html"

    # Sort by timestamp
    results = sorted(results, key=lambda r: r.get("timestamp", ""))

    # Calculate stats for template - use averages per model
    successful_results = [r for r in results if r.get("success")]

    # Group results by model and calculate averages
    model_stats = {}
    for r in successful_results:
        model_name = r.get("model_name")
        if not model_name:
            continue
        if model_name not in model_stats:
            model_stats[model_name] = {"f1": [], "time": [], "memory": []}
        if r.get("f1"):
            model_stats[model_name]["f1"].append(r["f1"])
        if r.get("time_sec"):
            model_stats[model_name]["time"].append(r["time_sec"])
        if r.get("memory_delta_gb", 0) > 0:
            model_stats[model_name]["memory"].append(r["memory_delta_gb"])

    # Calculate averages per model
    model_avgs = {}
    for model_name, stats in model_stats.items():
        model_avgs[model_name] = {
            "avg_f1": sum(stats["f1"]) / len(stats["f1"]) if stats["f1"] else 0,
            "avg_time": sum(stats["time"]) / len(stats["time"]) if stats["time"] else float("inf"),
            "avg_memory": sum(stats["memory"]) / len(stats["memory"]) if stats["memory"] else float("inf"),
        }

    # Best average F1
    best_f1_model = max(model_avgs.items(), key=lambda x: x[1]["avg_f1"], default=(None, {"avg_f1": 0}))
    best_f1 = best_f1_model[1]["avg_f1"] * 100
    best_f1_model_name = best_f1_model[0] if best_f1_model[0] else "-"

    # Fastest average model
    fastest_model_data = min(model_avgs.items(), key=lambda x: x[1]["avg_time"], default=(None, {"avg_time": 0}))
    fastest_time = fastest_model_data[1]["avg_time"] if fastest_model_data[0] else 0
    fastest_model = fastest_model_data[0] if fastest_model_data[0] else "-"

    # Lowest average memory model
    models_with_memory = {k: v for k, v in model_avgs.items() if v["avg_memory"] < float("inf")}
    lowest_mem_data = min(models_with_memory.items(), key=lambda x: x[1]["avg_memory"], default=(None, {"avg_memory": 0}))
    lowest_memory = round(lowest_mem_data[1]["avg_memory"], 1) if lowest_mem_data[0] else 0
    lowest_memory_model = lowest_mem_data[0] if lowest_mem_data[0] else "-"

    # Unique models
    models_tested = len(set(r.get("model_name") for r in results if r.get("model_name")))

    # Last run timestamp
    last_run = "-"
    if results:
        last_ts = results[-1].get("timestamp", "")
        if last_ts:
            last_run = last_ts[:16].replace("T", " ")

    # Get system info
    sys_info = get_system_info()

    # Load and render template
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html.j2")

    html_content = template.render(
        data_json=json.dumps(results, default=str),
        total_runs=len(results),
        best_f1=f"{best_f1:.1f}",
        best_f1_model=best_f1_model_name,
        fastest_time=f"{fastest_time:.1f}",
        fastest_model=fastest_model,
        lowest_memory=lowest_memory,
        lowest_memory_model=lowest_memory_model,
        models_tested=models_tested,
        last_run=last_run,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # System info
        os_version=sys_info["os_version"],
        cpu=sys_info["cpu"],
        ram_gb=sys_info["ram_gb"],
        gpu=sys_info["gpu"],
    )

    with open(report_file, "w") as f:
        f.write(html_content)

    # Auto-copy to docs/ for GitHub Pages
    docs_dir = Path(__file__).parent / "docs"
    docs_index = docs_dir / "index.html"
    if docs_dir.is_dir():
        import shutil
        shutil.copy2(report_file, docs_index)

    return report_file, docs_index if docs_dir.is_dir() else None


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load and validate config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Validated configuration dictionary

    Raises:
        ValueError: If required fields are missing or model configs are invalid
        FileNotFoundError: If config file doesn't exist
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    required = ["prompt", "models"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # Validate each model config
    if not isinstance(config["models"], list) or len(config["models"]) == 0:
        raise ValueError("Config 'models' must be a non-empty list")

    for i, model in enumerate(config["models"]):
        validate_model_config(model, i)

    if "settings" not in config:
        config["settings"] = {}

    defaults = {
        "min_free_ram_buffer_gb": 4,
        "auto_start_mlx_server": True,
        "mlx_server_port": 8080,
        "timeout_seconds": 600,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }

    for key, value in defaults.items():
        if key not in config["settings"]:
            config["settings"][key] = value

    return config


# =============================================================================
# Interactive Prompts
# =============================================================================

def prompt_model_selection(models: list[dict], settings: dict, results_dir: Path) -> Optional[dict]:
    """Interactively prompt user to select a model."""
    available_gb = get_available_memory_gb()
    buffer_gb = settings["min_free_ram_buffer_gb"]

    print("\nSelect a model to run:\n")

    for i, model in enumerate(models, 1):
        estimated = estimate_model_memory_gb(model)
        fit_indicator = get_memory_fit_indicator(estimated, available_gb, buffer_gb)

        display = get_display_name(model)
        last_run = get_last_run_time(results_dir, display)
        last_run_info = f" (last: {last_run})" if last_run else ""

        print(f"  [{i}] {display:<40} ({estimated:.0f} GB required)  {fit_indicator}{last_run_info}")

    print("\n  [0] Cancel\n")

    while True:
        try:
            choice = input("> ").strip()
            if choice == "0" or choice.lower() == "q":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")


# =============================================================================
# Single Model Benchmark
# =============================================================================

def run_single_benchmark(
    selected_model: dict,
    config: dict,
    settings: dict,
    results_dir: Path,
    ground_truth: Optional[dict],
    skip_warnings: bool = False,
) -> bool:
    """Run benchmark for a single model. Returns True if successful."""

    prompt_hash = save_prompt_if_new(results_dir, config["prompt"])

    print()
    print("=" * 60)
    print(f"Running: {get_display_name(selected_model)}")
    print(f"Output:  {results_dir}")
    print("=" * 60)

    # Memory check
    available_gb = get_available_memory_gb()
    estimated_gb = estimate_model_memory_gb(selected_model)
    buffer_gb = settings["min_free_ram_buffer_gb"]
    required_gb = estimated_gb + buffer_gb

    print(f"\nMemory check:")
    print(f"  Required: {required_gb:.1f} GB (model: {estimated_gb:.1f} + buffer: {buffer_gb:.1f})")
    print(f"  Available: {available_gb:.1f} GB")

    if available_gb < required_gb:
        shortfall = required_gb - available_gb
        print(f"  ⚠ Warning: {shortfall:.1f} GB short.")
        if not skip_warnings:
            response = input("\nContinue anyway? [y/N] ").strip().lower()
            if response != "y":
                return False
        else:
            print("  (--skip-warnings: continuing anyway)")
    else:
        print(f"  ✓ OK ({available_gb - required_gb:.1f} GB headroom)")

    # Track MLX process for cleanup
    mlx_proc = None

    # Capture memory BEFORE loading model
    memory_before = get_memory_pressure()
    print(f"\nMemory before: {memory_before['used_gb']:.1f} GB used")

    # Backend setup
    print()
    if selected_model["backend"] == "mlx":
        if settings["auto_start_mlx_server"]:
            mlx_proc = start_mlx_server(
                selected_model["model_id"], settings["mlx_server_port"]
            )
            if mlx_proc is None:
                print("\n✗ Failed to start MLX server.")
                return False
        elif not is_mlx_server_running(settings["mlx_server_port"]):
            print(f"✗ MLX server not running on port {settings['mlx_server_port']}")
            return False
        else:
            print(f"  Using existing MLX server on port {settings['mlx_server_port']}")

    elif selected_model["backend"] == "ollama":
        stop_mlx_server()
        if not is_ollama_running():
            print("⚠ Ollama not running. Start it with: ollama serve")
            if not skip_warnings:
                response = input("\nContinue anyway? [y/N] ").strip().lower()
                if response != "y":
                    return False
            else:
                print("  (--skip-warnings: continuing anyway)")
        else:
            print("  ✓ Ollama is running")

    # Run benchmark
    print(f"\nRunning benchmark (timeout: {settings['timeout_seconds']}s)...")
    print()

    benchmark_result = run_benchmark(
        backend=selected_model["backend"],
        model_id=selected_model["model_id"],
        prompt=config["prompt"],
        timeout=settings["timeout_seconds"],
        mlx_port=settings["mlx_server_port"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )

    memory_after = get_memory_pressure()
    print(f"\nMemory after: {memory_after['used_gb']:.1f} GB used")

    # Evaluate if ground truth provided
    evaluation = None
    if benchmark_result["success"] and ground_truth:
        print("\nEvaluating response...")
        predicted = extract_json_from_response(benchmark_result["response"])
        evaluation = evaluate_extraction(predicted, ground_truth)

        print(f"  JSON Valid: {'✓' if evaluation['json_valid'] else '✗'}")
        print(f"  Overall F1: {evaluation['overall_f1']:.1%}")
        print(f"  Precision:  {evaluation.get('overall_precision', 0):.1%}")
        print(f"  Recall:     {evaluation.get('overall_recall', 0):.1%}")

    if benchmark_result["success"]:
        print(f"\n✓ Completed in {benchmark_result['time_seconds']:.1f}s")
        if benchmark_result.get("total_tokens"):
            tokens_per_sec = benchmark_result["completion_tokens"] / benchmark_result["time_seconds"]
            print(f"  Tokens: {benchmark_result['total_tokens']} ({tokens_per_sec:.1f} tok/s)")
    else:
        print(f"\n✗ Failed: {benchmark_result.get('error', 'Unknown error')}")

    # Create timestamp for this run
    run_timestamp = datetime.now().isoformat()

    # Save response to file
    save_response(results_dir, selected_model, benchmark_result, run_timestamp)

    # Calculate memory delta
    memory_delta_gb = memory_after["used_gb"] - memory_before["used_gb"]

    # Build CSV row
    csv_row = {
        "timestamp": run_timestamp,
        "model_name": get_display_name(selected_model),
        "model_id": selected_model["model_id"],
        "backend": selected_model["backend"],
        "prompt_hash": prompt_hash,
        "prompt_preview": config["prompt"][:PROMPT_PREVIEW_LENGTH].replace("\n", " "),
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
        "time_sec": round(benchmark_result["time_seconds"], 2),
        "tokens": benchmark_result.get("total_tokens", ""),
        "prompt_tokens": benchmark_result.get("prompt_tokens", ""),
        "completion_tokens": benchmark_result.get("completion_tokens", ""),
        "memory_delta_gb": round(memory_delta_gb, 2),
        "f1": round(evaluation["overall_f1"], 4) if evaluation else "",
        "precision": round(evaluation.get("overall_precision", 0), 4) if evaluation else "",
        "recall": round(evaluation.get("overall_recall", 0), 4) if evaluation else "",
        "json_valid": evaluation["json_valid"] if evaluation else "",
        "success": benchmark_result["success"],
        "error": benchmark_result.get("error", ""),
    }

    # Append to CSV
    append_to_csv(results_dir, csv_row)
    print(f"\n✓ Results logged to {results_dir / CSV_FILENAME}")

    # Cleanup: stop MLX server or unload Ollama model
    if selected_model["backend"] == "mlx":
        stop_mlx_server(mlx_proc)
    elif selected_model["backend"] == "ollama":
        stop_ollama_model(selected_model["model_id"])

    # Summary of this run
    print()
    print("-" * 60)
    print(f"Model:    {get_display_name(selected_model)}")
    print(f"Backend:  {selected_model['backend']}")
    print(f"Time:     {benchmark_result['time_seconds']:.1f}s")
    print(f"Memory Δ: {memory_delta_gb:.1f} GB")
    if evaluation:
        print(f"F1 Score: {evaluation['overall_f1']:.1%}")

    return benchmark_result["success"]


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark local LLMs with structured extraction evaluation"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--model", "-m",
        type=int,
        metavar="N",
        help="Run model N (1-indexed) without interactive prompt",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models sequentially",
    )
    parser.add_argument(
        "--skip-warnings",
        action="store_true",
        help="Skip confirmation prompts for memory/process warnings",
    )

    args = parser.parse_args()

    if args.model and args.all:
        print("Error: Cannot use --model and --all together")
        sys.exit(1)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    settings = config["settings"]
    results_dir = Path("results")
    ground_truth = config.get("ground_truth")

    # Print header
    print("=" * 60)
    print("Local LLM Benchmark")
    print("=" * 60)
    print(f"System RAM: {get_system_memory_gb():.0f} GB")
    print(f"Available RAM: {get_available_memory_gb():.1f} GB")
    if ground_truth:
        print("Evaluation: ✓ Ground truth provided (F1 scoring enabled)")

    # Pre-check: alert if MLX or Ollama processes are already running
    pre_processes = check_llm_processes()
    if print_process_warning(pre_processes, "pre"):
        if not args.skip_warnings:
            response = input("\nContinue anyway? [y/N] ").strip().lower()
            if response != "y":
                sys.exit(0)
        else:
            print("  (--skip-warnings: continuing anyway)")

    # Determine which models to run
    models = config["models"]
    if args.model:
        if args.model < 1 or args.model > len(models):
            print(f"Error: --model must be between 1 and {len(models)}")
            sys.exit(1)
        models_to_run = [models[args.model - 1]]
    elif args.all:
        models_to_run = models
    else:
        # Interactive selection
        selected = prompt_model_selection(models, settings, results_dir)
        if not selected:
            print("Cancelled.")
            sys.exit(0)
        models_to_run = [selected]

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    successful = 0
    failed = 0

    for i, model in enumerate(models_to_run):
        if len(models_to_run) > 1:
            print()
            print("=" * 60)
            print(f"Model {i + 1} of {len(models_to_run)}")
            print("=" * 60)

        success = run_single_benchmark(
            selected_model=model,
            config=config,
            settings=settings,
            results_dir=results_dir,
            ground_truth=ground_truth,
            skip_warnings=args.skip_warnings,
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Generate final reports
    print()
    print("=" * 60)
    print("Generating reports...")
    report_file = generate_report(results_dir)
    if report_file:
        print(f"  Markdown: {report_file}")
    html_result = generate_html_report(results_dir)
    if html_result:
        html_report, docs_copy = html_result
        print(f"  HTML:     {html_report}")
        if docs_copy:
            print(f"  GitHub Pages: {docs_copy}")
        import webbrowser
        webbrowser.open(f"file://{html_report.resolve()}")

    # Final summary
    print()
    print("=" * 60)
    if len(models_to_run) > 1:
        print(f"Completed: {successful}/{len(models_to_run)} models")
        if failed > 0:
            print(f"Failed: {failed}")
    print(f"Results: {results_dir}")
    print("=" * 60)

    # Post-check: alert if MLX or Ollama processes are still running
    post_processes = check_llm_processes()
    print_process_warning(post_processes, "post")


if __name__ == "__main__":
    main()
