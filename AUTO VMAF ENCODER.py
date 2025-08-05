import configparser
import concurrent.futures
import hashlib
import queue
import select
import itertools
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
)
from rich.table import Table
from rich.text import Text


try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False




@dataclass
class EncodingSettings:
    
    ffmpeg_path: str; ffprobe_path: str; vmaf_model_path: str; database_path: str
    
    max_workers: int; num_parallel_vmaf_runs: int; max_iterations: int
    memory_limit_gb: float
    
    encoder_type: str
    
    nvenc_preset: str; nvenc_quality_mode: str
    nvenc_advanced_params: str
    
    svt_av1_preset: int; svt_av1_film_grain: int
    svt_av1_advanced_params: str
    
    target_vmaf: float; vmaf_tolerance: float
    cq_search_min: int; cq_search_max: int
    
    sampling_method: str
    sample_segment_duration: int; num_samples: int
    master_sample_encoder: str;
    min_scene_score: float
    min_scene_changes_required: int
    min_keyframes_required: int
    skip_start_seconds: int
    skip_end_seconds: int
    
    min_duration_seconds: int; min_filesize_mb: int
    min_bitrate_4k_kbps: int; min_bitrate_1080p_kbps: int; min_bitrate_720p_kbps: int
    
    enable_cache: bool; cache_database_path: str
    enable_performance_cache: bool; performance_database_path: str
    
    delete_source_file: bool; output_suffix: str; output_directory: str
    use_different_input_directory: bool; input_directory: str
    min_size_reduction_threshold: float
    rename_skipped_files: bool; skipped_file_suffix: str
    
    output_bit_depth: str

    skip_encoding_if_target_not_reached: bool

SETTINGS: EncodingSettings

def load_settings(config_file='config.ini'):
    """Load settings from config.ini and populate the SETTINGS object."""
    global SETTINGS
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        console.print(f"[bold red]Error: Configuration file '{config_file}' not found. Please create it.[/bold red]"); return False
    try:
        config.read(config_file, encoding='utf-8')
        cpu_count = psutil.cpu_count(logical=True)
        SETTINGS = EncodingSettings(
            
            ffmpeg_path=config.get('Paths', 'ffmpeg_path'),
            ffprobe_path=config.get('Paths', 'ffprobe_path'),
            vmaf_model_path=config.get('Paths', 'vmaf_model_path'),
            database_path=config.get('Paths', 'database_path'),
            
            max_workers=min(config.getint('Performance', 'max_workers', fallback=cpu_count), cpu_count),
            num_parallel_vmaf_runs=config.getint('Performance', 'num_parallel_vmaf_runs', fallback=3),
            max_iterations=config.getint('Performance', 'max_iterations', fallback=7),
            memory_limit_gb=config.getfloat('Performance', 'memory_limit_gb', fallback=8.0),
            
            encoder_type=config.get('Encoder', 'encoder_type', fallback='nvenc').lower(),
            
            nvenc_preset=config.get('NVENC', 'nvenc_preset', fallback='p4'),
            nvenc_quality_mode=config.get('NVENC', 'nvenc_quality_mode', fallback='quality'),
            nvenc_advanced_params=config.get('NVENC_Advanced', 'extra_params', fallback=''),
            
            svt_av1_preset=config.getint('SVT_AV1', 'svt_av1_preset', fallback=7),
            svt_av1_film_grain=config.getint('SVT_AV1', 'svt_av1_film_grain', fallback=0),
            svt_av1_advanced_params=config.get('SVT_AV1_Advanced', 'extra_params', fallback=''),
            
            target_vmaf=config.getfloat('VMAF_Targeting', 'target_vmaf'),
            vmaf_tolerance=config.getfloat('VMAF_Targeting', 'vmaf_tolerance'),
            cq_search_min=config.getint('VMAF_Targeting', 'cq_search_min'),
            cq_search_max=config.getint('VMAF_Targeting', 'cq_search_max'),
          
            
            sampling_method=config.get('VMAF_Sampling', 'sampling_method', fallback='tier0').lower(),
            sample_segment_duration=config.getint('VMAF_Sampling', 'sample_segment_duration'),
            num_samples=config.getint('VMAF_Sampling', 'num_samples'),
            master_sample_encoder=config.get('VMAF_Sampling', 'master_sample_encoder', fallback='software').lower(),
            min_scene_score=config.getfloat('VMAF_Sampling', 'min_scene_score', fallback=0.5),
            min_scene_changes_required=config.getint('VMAF_Sampling', 'min_scene_changes_required', fallback=5),
            min_keyframes_required=config.getint('VMAF_Sampling', 'min_keyframes_required', fallback=5),
            skip_start_seconds=config.getint('VMAF_Sampling', 'skip_start_seconds', fallback=0),
            skip_end_seconds=config.getint('VMAF_Sampling', 'skip_end_seconds', fallback=0),
            
            min_duration_seconds=config.getint('File_Filtering', 'min_duration_seconds', fallback=60),
            min_filesize_mb=config.getint('File_Filtering', 'min_filesize_mb', fallback=100),
            min_bitrate_4k_kbps=config.getint('File_Filtering', 'min_bitrate_4k_kbps', fallback=6000),
            min_bitrate_1080p_kbps=config.getint('File_Filtering', 'min_bitrate_1080p_kbps', fallback=2500),
            min_bitrate_720p_kbps=config.getint('File_Filtering', 'min_bitrate_720p_kbps', fallback=1500),
            
            enable_cache=config.getboolean('VMAF_Cache', 'enable_cache', fallback=True),
            cache_database_path=config.get('Paths', 'vmaf_cache_path', fallback='vmaf_cache.db'),
            enable_performance_cache=config.getboolean('Performance_Cache', 'enable_performance_cache', fallback=True),
            performance_database_path=config.get('Paths', 'performance_db_path', fallback='performance.db'),
            
            delete_source_file=config.getboolean('File_Management', 'delete_source_file', fallback=False),
            output_suffix=config.get('File_Management', 'output_suffix', fallback='_av1'),
            output_directory=config.get('File_Management', 'output_directory', fallback=''),
            use_different_input_directory=config.getboolean('File_Management', 'use_different_input_directory', fallback=False),
            input_directory=config.get('File_Management', 'input_directory', fallback=''),
            min_size_reduction_threshold=config.getfloat('File_Management', 'min_size_reduction_threshold', fallback=5.0),
            rename_skipped_files=config.getboolean('File_Management', 'rename_skipped_files', fallback=False),
            skipped_file_suffix=config.get('File_Management', 'skipped_file_suffix', fallback='_notencoded'),
            skip_encoding_if_target_not_reached=config.getboolean('File_Management', 'skip_encoding_if_target_not_reached', fallback=False),
            
            output_bit_depth=config.get('Output', 'output_bit_depth', fallback='source').lower()
        )
        
        if config.has_option('VMAF_Sampling', 'use_smart_sampling'):
            if not config.getboolean('VMAF_Sampling', 'use_smart_sampling'):
                SETTINGS.sampling_method = 'tier2'

        if SETTINGS.cq_search_min >= SETTINGS.cq_search_max:
            console.print(f"[bold red]Configuration error: cq_search_min ({SETTINGS.cq_search_min}) must be less than cq_search_max ({SETTINGS.cq_search_max})[/bold red]")
            return False


        return True
    except (configparser.Error, ValueError) as e:
        console.print(f"[bold red]Configuration error: {e}[/bold red]"); return False



def display_compact_summary(settings: EncodingSettings, console: Console):
    """
    Displays a compact, flush-left summary of the encoding settings.
    This version has all settings and precise spacing control.
    """

    def print_setting(key: str, value: Text | str):
        line = Text(style="dim")
        line.append(key)
        line.append(" ")
        if isinstance(value, str):
            line.append_text(Text.from_markup(value))
        else:
            line.append_text(value)
        console.print(line)

    console.print(Text("Encoder & Quality", style="bold cyan"))
    if settings.encoder_type == 'nvenc':
        encoder_info = f"NVENC ({settings.nvenc_preset}, {settings.nvenc_quality_mode.upper()})"
    else:
        encoder_info = f"SVT-AV1 (Preset {settings.svt_av1_preset})"
    print_setting("Encoder:", f"[green]{encoder_info}[/green]")
    print_setting("Target VMAF:", f"[green]{settings.target_vmaf} (±{settings.vmaf_tolerance})[/green]")
    print_setting("CQ/CRF Range:", f"[green]{settings.cq_search_min}-{settings.cq_search_max}[/green]")
    print_setting("Output Bit Depth:", f"[green]{settings.output_bit_depth.capitalize()}[/green]")

    console.print(Text("Performance & Caching", style="bold cyan"))
    vmaf_status = '[green]On[/green]' if settings.enable_cache else '[red]Off[/red]'
    perf_status = '[green]On[/green]' if settings.enable_performance_cache else '[red]Off[/red]'
    sampling_method_map = {'tier0': 'Tier0 (Scene Detect)', 'tier1': 'Tier1 (Keyframe)', 'tier2': 'Tier2 (Intervals)'}
    sampling_display_name = sampling_method_map.get(settings.sampling_method, settings.sampling_method.title())
    print_setting("Sampling Method:", f"[green]{sampling_display_name}[/green] ([green]{settings.num_samples}[/green] samples, [green]{settings.sample_segment_duration}[/green]s each)")
    print_setting("Workers:", f"[green]{settings.max_workers}[/green] parallel [white]file[/white](s), [green]{settings.num_parallel_vmaf_runs}[/green] parallel VMAF tests")
    print_setting("Caching:", Text.from_markup(f"VMAF {vmaf_status}, Performance {perf_status}"))

    console.print(Text("File Management & Filtering", style="bold cyan"))
    if settings.delete_source_file:
        print_setting("Source Files:", "[bold red]⚠️ DELETE after successful encoding[/bold red]")
    else:
        print_setting("Source Files:", f"[green]Keep[/green] and save with '[green]{settings.output_suffix}[/green]' suffix")
    if settings.min_size_reduction_threshold != 0:
        print_setting("Success Threshold:", f"Must save > [green]{settings.min_size_reduction_threshold}%[/green] file size")
    print_setting("Skip Suffix:", f"'[green]{settings.output_suffix}[/green]' and '[green]{settings.skipped_file_suffix}[/green]'")
    if settings.skip_encoding_if_target_not_reached:
        print_setting("On Failure:", "[green]Skip encoding if target VMAF not achievable[/green]")
    filter_parts = []
    if settings.min_duration_seconds > 0: filter_parts.append(f"< {settings.min_duration_seconds}s")
    if settings.min_filesize_mb > 0: filter_parts.append(f"< {settings.min_filesize_mb}MB")
    if filter_parts:
        print_setting("File Filtering:", f"Skip if {' or '.join(filter_parts)}")
    bitrate_parts = []
    if settings.min_bitrate_4k_kbps > 0: bitrate_parts.append(f"4K: {settings.min_bitrate_4k_kbps}")
    if settings.min_bitrate_1080p_kbps > 0: bitrate_parts.append(f"1080p: {settings.min_bitrate_1080p_kbps}")
    if settings.min_bitrate_720p_kbps > 0: bitrate_parts.append(f"720p: {settings.min_bitrate_720p_kbps}")
    if bitrate_parts:
        print_setting("Bitrate Filtering:", f"Skip below {', '.join(bitrate_parts)} kbps")



console = Console()
not_replaced_files = []
worker_logs = {}
worker_logs_lock = threading.Lock()
FFMPEG_ENV = {}

class MemoryManager:
    """Manages memory usage to prevent crashes, tracking all FFmpeg processes."""
    def __init__(self, limit_gb: float):
        self.limit_gb = limit_gb
        self.main_process = psutil.Process(os.getpid())

    def get_all_ffmpeg_processes(self):
        """Get all FFmpeg processes spawned by this script."""
        try:
            children = self.main_process.children(recursive=True)
            ffmpeg_processes = [p for p in children if 'ffmpeg' in p.name().lower()]
            return ffmpeg_processes
        except:
            return []


    def get_total_memory_usage(self) -> int:
        """Get total memory usage of main process + all FFmpeg children."""
        try:
            total = self.main_process.memory_info().rss
            for ffmpeg_proc in self.get_all_ffmpeg_processes():
                try:
                    total += ffmpeg_proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return total
        except:
            return self.main_process.memory_info().rss

    def get_usage_mb(self) -> float:
        return self.get_total_memory_usage() / (1024**2)

    def get_detailed_usage(self) -> Dict:
        """Get detailed memory breakdown."""
        try:
            main_memory = self.main_process.memory_info().rss
            ffmpeg_processes = self.get_all_ffmpeg_processes()
            ffmpeg_memory = sum(p.memory_info().rss for p in ffmpeg_processes if p.is_running())

            return {
                'main_mb': main_memory / (1024**2),
                'ffmpeg_mb': ffmpeg_memory / (1024**2),
                'total_mb': (main_memory + ffmpeg_memory) / (1024**2),
                'ffmpeg_count': len(ffmpeg_processes)
            }
        except:
            total = self.get_total_memory_usage()
            return {
                'main_mb': total / (1024**2),
                'ffmpeg_mb': 0,
                'total_mb': total / (1024**2),
                'ffmpeg_count': 0
            }

class VMAFCache:
    """Caches VMAF results to avoid re-calculating for the same video samples."""
    def __init__(self, db_path: str):
        self.db_path = db_path; self._local = threading.local()
        self._source_file_hashes = {}; self._source_file_hashes_lock = threading.Lock()
        self._get_conn()
    def _get_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.execute('CREATE TABLE IF NOT EXISTS vmaf_cache (sample_key TEXT, cq INTEGER, vmaf_score REAL, timestamp REAL, PRIMARY KEY (sample_key, cq))')
        return self._local.conn
    def _get_source_file_hash(self, file_path: str) -> str:
        with self._source_file_hashes_lock:
            if file_path in self._source_file_hashes: return self._source_file_hashes[file_path]
            h = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192*4): h.update(chunk)
            file_hash = h.hexdigest()
            self._source_file_hashes[file_path] = file_hash
            return file_hash
    def _get_sample_key(self, source_video_path: str, sample_timestamps: List[float], complexity_score: float = None) -> str:
        source_hash = self._get_source_file_hash(source_video_path)
        timestamps_str = str(sorted(sample_timestamps))
        
        if complexity_score is not None:
            
            complexity_str = f"-c{complexity_score:.1f}"
        else:
            complexity_str = ""
        combined_string = f"{source_hash}-{timestamps_str}{complexity_str}"
        return hashlib.sha256(combined_string.encode()).hexdigest()
    def get(self, source_video_path: str, sample_timestamps: List[float], cq: int, complexity_score: float = None) -> Optional[float]:
        conn = self._get_conn()
        sample_key = self._get_sample_key(source_video_path, sample_timestamps, complexity_score)
        cursor = conn.execute('SELECT vmaf_score FROM vmaf_cache WHERE sample_key=? AND cq=?', (sample_key, cq))
        result = cursor.fetchone()
        return result[0] if result else None

    def set(self, source_video_path: str, sample_timestamps: List[float], cq: int, vmaf_score: float, complexity_score: float = None):
        conn = self._get_conn()
        sample_key = self._get_sample_key(source_video_path, sample_timestamps, complexity_score)
        conn.execute('INSERT OR REPLACE INTO vmaf_cache VALUES (?, ?, ?, ?)', (sample_key, cq, vmaf_score, time.time()))
        conn.commit()

class PerformanceDB:
    """Logs and retrieves performance data to predict ETA."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._get_conn()

    def _get_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_log (
                    resolution_key TEXT,
                    encoder_type TEXT,
                    preset TEXT,
                    sample_creation_time REAL,
                    vmaf_search_time REAL,
                    final_encode_fps REAL,
                    timestamp REAL
                )
            ''')
            
            migrations = [
                'ALTER TABLE performance_log ADD COLUMN encoder_type TEXT',
                'ALTER TABLE performance_log ADD COLUMN complexity_score REAL DEFAULT 0.5',
                'ALTER TABLE performance_log ADD COLUMN scene_count INTEGER DEFAULT 0',
                'ALTER TABLE performance_log ADD COLUMN sampling_method TEXT DEFAULT "unknown"'
            ]
            for migration in migrations:
                try:
                    self._local.conn.execute(migration)
                    self._local.conn.commit()
                except sqlite3.OperationalError:
                    pass 

            
            try:
                self._local.conn.execute('CREATE INDEX IF NOT EXISTS idx_complexity ON performance_log(resolution_key, encoder_type, preset, complexity_score)')
                self._local.conn.commit()
            except sqlite3.OperationalError:
                pass
        return self._local.conn

    def log_performance(self, metrics: Dict, timings: Dict):
        """Logs performance metrics for a completed encode, including complexity data."""
        conn = self._get_conn()
        video_stream = next((s for s in metrics.get('media_info', {}).get('streams', []) if s.get('codec_type') == 'video'), None)
        if not video_stream: return

        resolution_key = classify_resolution(video_stream.get('width'), video_stream.get('height'))

        if SETTINGS.encoder_type == 'nvenc':
            active_preset = SETTINGS.nvenc_preset
        else:
            active_preset = str(SETTINGS.svt_av1_preset)

        
        complexity_data = metrics.get('complexity_data', {})
        complexity_score = complexity_data.get('complexity_score', 0.5)
        scene_count = complexity_data.get('scene_count', 0)
        sampling_method = complexity_data.get('sampling_method', 'unknown')

        conn.execute('INSERT INTO performance_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (
            resolution_key, SETTINGS.encoder_type, active_preset,
            timings.get('sample_creation_time', 0),
            timings.get('vmaf_search_time', 0),
            timings.get('final_encode_fps', 0),
            time.time(),
            complexity_score,
            scene_count,
            sampling_method
        ))
        conn.commit()

    def get_component_stats(self, resolution_key: str, encoder_type: str, preset: str, complexity_score: float = None) -> Dict:
        """Get statistics for each performance component, filtered by encoder and optionally by complexity."""
        conn = self._get_conn()

        
        if complexity_score is not None:
            
            complexity_range = 0.2
            query_base = 'FROM performance_log WHERE resolution_key=? AND encoder_type=? AND preset=? AND complexity_score BETWEEN ? AND ?'
            params = (resolution_key, encoder_type, preset,
                      max(0, complexity_score - complexity_range),
                      min(1, complexity_score + complexity_range))
        else:
            
            query_base = 'FROM performance_log WHERE resolution_key=? AND encoder_type=? AND preset=?'
            params = (resolution_key, encoder_type, preset)

        cursor = conn.execute(f'SELECT AVG(sample_creation_time), COUNT(*) {query_base} AND sample_creation_time > 0', params)
        sample_result = cursor.fetchone()

        cursor = conn.execute(f'SELECT AVG(vmaf_search_time), COUNT(*) {query_base} AND vmaf_search_time > 0', params)
        vmaf_result = cursor.fetchone()

        cursor = conn.execute(f'SELECT AVG(final_encode_fps), COUNT(*) {query_base} AND final_encode_fps > 0', params)
        encode_result = cursor.fetchone()

        
        min_samples_needed = 3
        if complexity_score is not None and encode_result[1] < min_samples_needed:
            
            return self.get_component_stats(resolution_key, encoder_type, preset, complexity_score=None)

        return {
            'sample_avg': sample_result[0] if sample_result and sample_result[0] else 0,
            'sample_count': sample_result[1] if sample_result and sample_result[1] else 0,
            'vmaf_avg': vmaf_result[0] if vmaf_result and vmaf_result[0] else 0,
            'vmaf_count': vmaf_result[1] if vmaf_result and vmaf_result[1] else 0,
            'encode_fps': encode_result[0] if encode_result and encode_result[0] else 0,
            'encode_count': encode_result[1] if encode_result and encode_result[1] else 0
        }

    def get_complexity_adjusted_eta(self, metrics: Dict, complexity_data: Dict = None) -> float:
        video_stream = next((s for s in metrics.get('media_info', {}).get('streams', [])
                             if s.get('codec_type') == 'video'), None)
        if not video_stream:
            return 300 

        resolution_key = classify_resolution(video_stream.get('width'), video_stream.get('height'))
        video_duration = metrics.get('video_duration_seconds', 0)

        framerate_str = video_stream.get('avg_frame_rate', '30/1')
        num, den = framerate_str.split('/') if '/' in framerate_str else (framerate_str, 1)
        framerate = float(num) / float(den) if float(den) > 0 else 30

        if SETTINGS.encoder_type == 'nvenc':
            active_preset = SETTINGS.nvenc_preset
        else:
            active_preset = str(SETTINGS.svt_av1_preset)

        complexity_score = None
        if complexity_data and 'complexity_score' in complexity_data:
            complexity_score = complexity_data['complexity_score']

        components = self.get_component_stats(resolution_key, SETTINGS.encoder_type,
                                              active_preset, complexity_score)

        eta = self.calculate_weighted_eta(components, video_duration, framerate, resolution_key)

        if complexity_score is not None:
            if complexity_score > 0.7:
                eta *= 1.2
            elif complexity_score < 0.3:
                eta *= 0.9

        return eta

    def get_conservative_vmaf_estimate(self, resolution_key: str) -> float:
        """Fallback VMAF time estimates based on resolution."""
        estimates = {'4K': 120.0, '1080p': 60.0, '720p': 30.0, 'SD': 20.0}
        return estimates.get(resolution_key, 60.0)

    def calculate_weighted_eta(self, components: Dict, video_duration: float, framerate: float, resolution_key: str) -> float:
        """Calculate ETA using advanced weighting with confidence factors."""
        sample_confidence = min(1.0, components['sample_count'] / 10.0)
        vmaf_confidence = min(1.0, components['vmaf_count'] / 5.0)
        encode_confidence = min(1.0, components['encode_count'] / 10.0)

        sample_time = components['sample_avg'] * (1.0 + (1.0 - sample_confidence) * 0.2)

        if components['vmaf_avg'] > 0 and vmaf_confidence > 0.2:
            vmaf_time = (components['vmaf_avg'] / SETTINGS.num_parallel_vmaf_runs) * (1.0 + (1.0 - vmaf_confidence) * 0.5)
        else:
            vmaf_time = self.get_conservative_vmaf_estimate(resolution_key)

        total_frames = video_duration * framerate
        if components['encode_fps'] > 0:
            encode_time = (total_frames / components['encode_fps']) * (1.0 + (1.0 - encode_confidence) * 0.3)
        else:
            encode_time = video_duration * 2.0 

        return max(sample_time + vmaf_time + encode_time, video_duration * 0.1)

    def predict_eta(self, metrics: Dict) -> float:
        video_stream = next((s for s in metrics.get('media_info', {}).get('streams', []) if s.get('codec_type') == 'video'), None)
        if not video_stream: return 300

        resolution_key = classify_resolution(video_stream.get('width'), video_stream.get('height'))
        video_duration = metrics.get('video_duration_seconds', 0)

        framerate_str = video_stream.get('avg_frame_rate', '30/1')
        num, den = framerate_str.split('/') if '/' in framerate_str else (framerate_str, 1)
        framerate = float(num) / float(den) if float(den) > 0 else 30

        if SETTINGS.encoder_type == 'nvenc':
            active_preset = SETTINGS.nvenc_preset
        else:
            active_preset = str(SETTINGS.svt_av1_preset)

        components = self.get_component_stats(resolution_key, SETTINGS.encoder_type, active_preset)
        eta = self.calculate_weighted_eta(components, video_duration, framerate, resolution_key)
        return eta

memory_manager: MemoryManager
vmaf_cache: VMAFCache
performance_db: PerformanceDB

def get_ffmpeg_env():
    """Sets up the environment for FFmpeg to find necessary libraries."""
    env = os.environ.copy()
    ffmpeg_bin_dir = Path(SETTINGS.ffmpeg_path).parent
    env['PATH'] = f"{ffmpeg_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env

def get_media_info(video_path: str) -> Optional[Dict]:
    """Retrieves media information using ffprobe."""
    if not os.path.exists(video_path): return None
    cmd = [
        str(SETTINGS.ffprobe_path),
        '-v', 'error',
        '-show_entries',
        'format=duration,bit_rate:stream=codec_name,codec_type,width,height,avg_frame_rate,pix_fmt,color_space,color_primaries,color_transfer',
        '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=False, env=FFMPEG_ENV, timeout=30)
        if result.returncode == 0 and result.stdout: return json.loads(result.stdout)
    except Exception: pass
    return None

def build_color_args(video_stream: Dict) -> List[str]:
    """Builds FFmpeg color-related arguments from a video stream's info."""
    args = []
    if video_stream.get('color_space') and video_stream['color_space'] != 'unknown':
        args.extend(['-colorspace', video_stream['color_space']])
    if video_stream.get('color_primaries') and video_stream['color_primaries'] != 'unknown':
        args.extend(['-color_primaries', video_stream['color_primaries']])
    if video_stream.get('color_trc') and video_stream['color_trc'] != 'unknown':
        args.extend(['-color_trc', video_stream['color_trc']])
    return args

def get_target_pix_fmt(source_pix_fmt: str, encoder_type: str, output_bit_depth: str) -> Optional[str]:
    """Determines the target pixel format based on user settings and encoder capabilities."""
    if not source_pix_fmt:
        return None

    is_10bit_source = '10' in source_pix_fmt or '12' in source_pix_fmt
    is_422_source = '422' in source_pix_fmt
    is_444_source = '444' in source_pix_fmt

    if output_bit_depth == 'source':
        return 'p010le' if encoder_type == 'nvenc' and source_pix_fmt == 'yuv420p10le' else source_pix_fmt

    if output_bit_depth == '8bit':
        if is_444_source: return 'yuv444p'
        if is_422_source: return 'yuv422p'
        return 'yuv420p'

    if output_bit_depth == '10bit':
        if is_10bit_source:
            return 'p010le' if encoder_type == 'nvenc' and '420' in source_pix_fmt else source_pix_fmt
        else:
            if encoder_type == 'nvenc':
                return 'p010le'
            else:
                if is_444_source: return 'yuv444p10le'
                if is_422_source: return 'yuv422p10le'
                return 'yuv420p10le'

    return source_pix_fmt

def build_encoder_args(quality_value: int, video_stream: Dict, for_final_encode: bool) -> List[str]:
    """Builds a list of FFmpeg arguments based on the selected encoder and color settings."""
    source_pix_fmt = video_stream.get('pix_fmt')
    color_args = build_color_args(video_stream)

    if for_final_encode:
        target_pix_fmt = get_target_pix_fmt(source_pix_fmt, SETTINGS.encoder_type, SETTINGS.output_bit_depth)
    else:
        target_pix_fmt = 'p010le' if SETTINGS.encoder_type == 'nvenc' and source_pix_fmt == 'yuv420p10le' else source_pix_fmt

    base_args = []
    if SETTINGS.encoder_type == 'nvenc':
        base_args = ['-c:v', 'av1_nvenc', '-preset', SETTINGS.nvenc_preset, '-rc', 'vbr', '-cq', str(quality_value), '-b:v', '0']
        if for_final_encode and SETTINGS.nvenc_advanced_params:
            base_args.extend(SETTINGS.nvenc_advanced_params.split())

    elif SETTINGS.encoder_type == 'svt_av1':
        base_args = ['-c:v', 'libsvtav1', '-preset', str(SETTINGS.svt_av1_preset), '-crf', str(quality_value), '-svtav1-params', f'film-grain={SETTINGS.svt_av1_film_grain}']
        if for_final_encode and SETTINGS.svt_av1_advanced_params:
            base_args.extend(SETTINGS.svt_av1_advanced_params.split())
    else:
        raise ValueError(f"Unsupported encoder type in config: {SETTINGS.encoder_type}")

    if target_pix_fmt:
        base_args.extend(['-pix_fmt', target_pix_fmt])
    base_args.extend(color_args)
    return base_args

def format_duration(seconds: float) -> str:
    """Formats seconds into a human-readable string (e.g., 1h 23m 45s)."""
    if seconds < 0: seconds = 0
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0: return f"{hours}h {mins}m {secs}s"
    if mins > 0: return f"{mins}m {secs}s"
    return f"{secs}s"

def get_file_size_info(filepath: str) -> tuple[int, float]:
    """Returns the file size in bytes and megabytes."""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes, size_bytes / (1024*1024)
    return 0, 0.0

def classify_resolution(width: Optional[int], height: Optional[int]) -> str:
    """Classifies video resolution into a string category (4K, 1080p, etc.)."""
    if not width or not height: return "1080p"
    if width >= 3840: return "4K"
    if width >= 1920: return "1080p"
    if width >= 1280: return "720p"
    return "SD"

def log_to_worker(task_id: int, message):
    """Logs a message to a specific worker's log buffer for display."""
    with worker_logs_lock:
        worker_logs.setdefault(task_id, []).append(message)
        worker_logs[task_id] = worker_logs[task_id][-15:]

def make_summary_panel(batch_state, progress):
    """Creates the main summary panel for the live display."""
    elapsed = time.time() - batch_state['start_time']
    space_saved_mb = (batch_state['deleted_source_size'] - batch_state['encoded_output_size']) / (1024*1024)
    memory_info = memory_manager.get_detailed_usage()

    summary_group = Table.grid(expand=True); summary_group.add_row(progress)
    stats_grid = Table.grid(expand=True); stats_grid.add_column(); stats_grid.add_column(justify="right")

    if batch_state['files_completed'] < batch_state['total_files']:
        eta_seconds = get_countdown_eta(batch_state)
        eta_str = f"ETA: {format_duration(eta_seconds)}" if eta_seconds > 0 else ""
        elapsed_str = f"Elapsed: {format_duration(elapsed)}"
        right_text = Text(f"{eta_str}\n{elapsed_str}", justify="right") if eta_str else Text(elapsed_str, justify="right")
    else:
        # When completed, only show elapsed time
        elapsed_str = f"Elapsed: {format_duration(elapsed)}"
        right_text = Text(f"Completed!\n{elapsed_str}", justify="right", style="green")

# Calculate total queue size display
    total_queue_size = batch_state.get('total_queue_size_bytes', 0)
    if total_queue_size >= 1024**3:  # >= 1GB
        queue_size_str = f"{total_queue_size / (1024**3):.1f}GB"
    elif total_queue_size >= 1024**2:  # >= 1MB
        queue_size_str = f"{total_queue_size / (1024**2):.0f}MB"
    else:
        queue_size_str = f"{total_queue_size / 1024:.0f}KB"

    # Calculate space saved percentage
    space_saved_bytes = batch_state['deleted_source_size'] - batch_state['encoded_output_size']
    space_saved_mb = space_saved_bytes / (1024*1024)
    if batch_state['deleted_source_size'] > 0:
        space_saved_percent = (space_saved_bytes / batch_state['deleted_source_size']) * 100
        space_saved_display = f"Space saved: {space_saved_mb:.2f} MB ({space_saved_percent:.1f}%)"
    else:
        space_saved_display = f"Space saved: {space_saved_mb:.2f} MB"

    left_column = Text()
    left_column.append(f"Files completed: {batch_state['files_completed']}/{batch_state['total_files']}\n")
    left_column.append(f"Total queue size: {queue_size_str}\n")
    left_column.append(space_saved_display)
     
    right_column = Text()
    right_column.append(right_text.plain + "\n")
    right_column.append(f"RAM Usage: {memory_info['total_mb']:.1f} MB")
    
    stats_grid.add_row(left_column, right_column)

    summary_group.add_row(stats_grid)
    return Panel(summary_group, title="[bold cyan]Encoding Summary", border_style="cyan", padding=(1, 2))

def generate_worker_panels(active_threads: list, progress_objects: dict):
    """Generates the individual panels for each active worker thread."""
    panels = []
    for task_id, thread, filename in active_threads:
        log_content = worker_logs.get(task_id, ["Initializing..."])
        panel_items = [Group(*log_content)]
        if task_id in progress_objects:
            panel_items.extend(progress_objects[task_id].values())
        panels.append(Panel(Group(*panel_items), title=f"[bold yellow]Worker:[/bold yellow] {os.path.basename(filename)}", border_style="yellow"))
    for _ in range(SETTINGS.max_workers - len(panels)):
        panels.append(Group())
    return Group(*panels)

def generate_layout(batch_state, total_progress, active_threads, worker_progress_objects):
    """Combines all UI components into a single layout for the Live display."""
    return Group(make_summary_panel(batch_state, total_progress), generate_worker_panels(active_threads, worker_progress_objects))


def test_enhanced_setup() -> bool:
    """Tests if FFmpeg, FFprobe, and VMAF are configured correctly."""
    for tool_name, tool_path in [("FFmpeg", SETTINGS.ffmpeg_path), ("FFprobe", SETTINGS.ffprobe_path)]:
        if not os.path.exists(tool_path):
            console.print(f"[red]{tool_name} not found: {tool_path}[/red]")
            return False
        try:
            if subprocess.run([tool_path, '-version'], capture_output=True, timeout=10).returncode != 0:
                console.print(f"[red]{tool_name} failed to run[/red]")
                return False
        except Exception as e:
            console.print(f"[red]{tool_name} test failed: {e}[/red]")
            return False

    if not os.path.exists(SETTINGS.vmaf_model_path):
        console.print(f"[red]VMAF model not found: {SETTINGS.vmaf_model_path}[/red]")
        return False

    model_path = SETTINGS.vmaf_model_path.replace('\\', '/')
    test_cmd = [
        SETTINGS.ffmpeg_path,
        '-f', 'lavfi',
        '-i', 'testsrc=duration=1:size=320x240:rate=1',
        '-f', 'lavfi',
        '-i', 'testsrc=duration=1:size=320x240:rate=1',
        '-lavfi', f'[0:v][1:v]libvmaf={model_path}:log_path=NUL',
        '-f', 'null', '-'
    ]

    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30, env=FFMPEG_ENV)
        if 'VMAF score:' not in result.stderr:
            console.print("[red]VMAF test failed - no score found[/red]")
            console.print(f"[dim]Output: {result.stderr[-500:]}[/dim]")
            return False
    except Exception as e:
        console.print(f"[red]VMAF test error: {e}[/red]")
        return False

    return True

def run_vmaf_comparison(encoded_path: str, reference_path: str, progress_callback=None) -> float:
    model_path = SETTINGS.vmaf_model_path.replace('\\', '/')
    filter_string = f'[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];[dist][ref]libvmaf={model_path}:log_path=NUL'
    cmd = [SETTINGS.ffmpeg_path, '-i', encoded_path, '-i', reference_path, '-lavfi', filter_string, '-f', 'null', '-']

    full_output = ""
    try:
        ref_duration_info = get_media_info(reference_path)
        ref_duration = float(ref_duration_info['format']['duration']) if ref_duration_info else 0
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, universal_newlines=True, encoding='utf-8', env=FFMPEG_ENV)
        time_pattern, last_progress = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})"), 0

        for line in process.stderr:
            full_output += line
            if progress_callback and ref_duration > 0:
                match = time_pattern.search(line)
                if match:
                    h, m, s, ms = map(int, match.groups())
                    current_seconds = h * 3600 + m * 60 + s + ms / 100.0
                    progress = min(100.0, (current_seconds / ref_duration) * 100)
                    if progress > last_progress + 1.0: 
                        last_progress = progress
                        progress_callback(progress)
            elif progress_callback:
                
                if 'frame=' in line:
                    progress_callback(min(90.0, last_progress + 0.5))

        process.wait()
        if progress_callback:
            progress_callback(100)

    except Exception as e:
        console.print(f"[red]Error during VMAF comparison: {e}[/red]")

    vmaf_match = re.search(r'VMAF score: ([\d\.]+)', full_output)
    return float(vmaf_match.group(1)) if vmaf_match else 0.0



def get_tier0_samples(input_path: str, log_callback, task_id: int, video_duration: float) -> Optional[Tuple[List[float], Dict[str, float]]]:
    """Tier 0: Uses PySceneDetect library to find scene changes and analyze complexity."""
    try:
        video = open_video(input_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=SETTINGS.min_scene_score))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        if len(scene_list) < SETTINGS.min_scene_changes_required:
            log_callback(task_id, f"Tier 0: Found only {len(scene_list)} scenes, need {SETTINGS.min_scene_changes_required}.")
            return None

        timestamps = sorted([scene[0].get_seconds() for scene in scene_list])

        
        scene_durations = []
        for i in range(len(scene_list)):
            start_time = scene_list[i][0].get_seconds()
            end_time = scene_list[i][1].get_seconds()
            scene_durations.append(end_time - start_time)

        
        avg_scene_duration = sum(scene_durations) / len(scene_durations) if scene_durations else 0
        min_scene_duration = min(scene_durations) if scene_durations else 0
        scenes_per_minute = (len(scene_list) / video_duration) * 60 if video_duration > 0 else 0

        
        quick_cuts = sum(1 for d in scene_durations if d < 2.0)
        quick_cut_ratio = quick_cuts / len(scene_durations) if scene_durations else 0

        
        
        complexity_score = min(1.0, (scenes_per_minute / 30.0) * 0.5 + quick_cut_ratio * 0.5)

        complexity_data = {
            'scene_count': len(scene_list),
            'avg_scene_duration': avg_scene_duration,
            'min_scene_duration': min_scene_duration,
            'scenes_per_minute': scenes_per_minute,
            'quick_cut_ratio': quick_cut_ratio,
            'complexity_score': complexity_score,
            'sampling_method': 'tier0'
        }

        log_callback(task_id, f"Tier 0: Detected {len(scene_list)} scenes, complexity score: {complexity_score:.2f}")

        if len(timestamps) > SETTINGS.num_samples:
            indices = [int(i * (len(timestamps) - 1) / (SETTINGS.num_samples - 1)) for i in range(SETTINGS.num_samples)]
            selected = sorted([timestamps[i] for i in set(indices)])
        else:
            selected = timestamps

        log_callback(task_id, f"Tier 0: Selected {len(selected)} samples using PySceneDetect.")
        return selected, complexity_data
    except Exception as e:
        log_callback(task_id, f"[red]Tier 0 Error: {e}[/red]")
        return None

def run_vmaf_comparison_in_memory(encoded_data: bytes, reference_data: bytes, progress_callback=None) -> float:
    """Runs VMAF comparison using in-memory data via temporary files with improved progress tracking."""
    model_path = SETTINGS.vmaf_model_path.replace('\\', '/')
    filter_string = f'[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];[dist][ref]libvmaf={model_path}:log_path=NUL'

    
    enc_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mkv', prefix='vmaf_enc_')
    ref_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mkv', prefix='vmaf_ref_')

    try:
        
        enc_temp.write(encoded_data)
        enc_temp.flush()
        enc_temp.close()

        ref_temp.write(reference_data)
        ref_temp.flush()
        ref_temp.close()

        
        if not (os.path.exists(enc_temp.name) and os.path.exists(ref_temp.name)):
            raise RuntimeError("Failed to create temporary files for VMAF comparison")

        if os.path.getsize(enc_temp.name) == 0 or os.path.getsize(ref_temp.name) == 0:
            raise RuntimeError("Temporary files are empty")

        cmd = [
            SETTINGS.ffmpeg_path,
            '-i', enc_temp.name,
            '-i', ref_temp.name,
            '-lavfi', filter_string,
            '-f', 'null', '-'
        ]

        
        ref_info_cmd = [
            SETTINGS.ffprobe_path, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0', ref_temp.name
        ]

        ref_duration = 0
        try:
            duration_result = subprocess.run(
                ref_info_cmd,
                capture_output=True,
                text=True,
                env=FFMPEG_ENV,
                timeout=15
            )
            if duration_result.returncode == 0 and duration_result.stdout.strip():
                ref_duration = float(duration_result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, subprocess.SubprocessError):
            ref_duration = 0

        
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            universal_newlines=True,
            encoding='utf-8',
            env=FFMPEG_ENV,
            bufsize=1
        )

        full_output = ""
        time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
        frame_pattern = re.compile(r"frame=\s*(\d+)")
        last_reported_progress = 0.0

        try:
            for line in process.stderr:
                full_output += line

                if progress_callback:
                    progress_updated = False

                    
                    if ref_duration > 0:
                        match = time_pattern.search(line)
                        if match:
                            h, m, s, ms = map(int, match.groups())
                            current_seconds = h * 3600 + m * 60 + s + ms / 100.0
                            progress = min(100.0, (current_seconds / ref_duration) * 100)

                            
                            if progress >= last_reported_progress + 0.5 or progress >= 100.0:
                                last_reported_progress = progress
                                progress_callback(progress)
                                progress_updated = True

                    
                    if not progress_updated:
                        frame_match = frame_pattern.search(line)
                        if frame_match:
                            current_frame = int(frame_match.group(1))
                            
                            estimated_total_frames = ref_duration * 30 if ref_duration > 0 else SETTINGS.sample_segment_duration * SETTINGS.num_samples * 30
                            if estimated_total_frames > 0:
                                frame_progress = min(95.0, (current_frame / estimated_total_frames) * 100)
                                if frame_progress > last_reported_progress:
                                    last_reported_progress = frame_progress
                                    progress_callback(frame_progress)
                                    progress_updated = True

                    
                    if not progress_updated and ('frame=' in line or 'fps=' in line or 'bitrate=' in line):
                        if last_reported_progress < 90.0:
                            last_reported_progress = min(90.0, last_reported_progress + 0.3)
                            progress_callback(last_reported_progress)

            process.wait(timeout=1500)

            
            if progress_callback:
                progress_callback(100.0)

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("VMAF comparison timed out after 5 minutes")

        if process.returncode != 0:
            error_msg = full_output[-1000:] if full_output else "Unknown FFmpeg error"
            raise RuntimeError(f"VMAF comparison failed with return code {process.returncode}: {error_msg}")

        
        vmaf_match = re.search(r'VMAF score: ([\d\.]+)', full_output)
        if not vmaf_match:
            
            alt_patterns = [
                r'vmaf=([0-9]+\.?[0-9]*)',
                r'VMAF.*?([0-9]+\.?[0-9]+)',
                r'mean: ([0-9]+\.?[0-9]+)'
            ]

            for pattern in alt_patterns:
                alt_match = re.search(pattern, full_output)
                if alt_match:
                    vmaf_match = alt_match
                    break

            if not vmaf_match:
                raise RuntimeError(f"No VMAF score found in output. Last 500 chars: {full_output[-500:]}")

        score = float(vmaf_match.group(1))

        
        if not (0.0 <= score <= 100.0):
            raise RuntimeError(f"Invalid VMAF score: {score}")

        return score

    except Exception as e:
        if progress_callback:
            progress_callback(100.0) 
        raise RuntimeError(f"VMAF comparison failed: {str(e)}")
    finally:
        
        for temp_file in [enc_temp.name, ref_temp.name]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except (OSError, PermissionError):
                pass 

def encode_sample_in_memory(reference_data: bytes, cq: int, video_stream: Dict) -> bytes:
    """Encodes a sample in memory using temporary files, preserving color information."""
    encoder_args = build_encoder_args(cq, video_stream, for_final_encode=False)

    
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mkv', prefix='enc_input_')
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mkv', prefix='enc_output_')

    try:
        
        input_temp.write(reference_data)
        input_temp.flush()
        input_temp.close()
        output_temp.close()

        
        if not os.path.exists(input_temp.name) or os.path.getsize(input_temp.name) == 0:
            raise RuntimeError("Failed to create valid input temporary file")

        cmd = [
            SETTINGS.ffmpeg_path,
            '-v', 'error', 
            '-i', input_temp.name,
            *encoder_args,
            '-y', output_temp.name
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=FFMPEG_ENV,
            timeout=1500, 
            encoding='utf-8'
        )

        if result.returncode != 0:
            error_msg = result.stderr[-1000:] if result.stderr else "Unknown encoding error"
            raise RuntimeError(f"In-memory encoding failed with return code {result.returncode}: {error_msg}")

        
        if not os.path.exists(output_temp.name):
            raise RuntimeError("Encoded output file was not created")

        output_size = os.path.getsize(output_temp.name)
        if output_size == 0:
            raise RuntimeError("Encoded output file is empty")

        
        with open(output_temp.name, 'rb') as f:
            encoded_data = f.read()

        if len(encoded_data) == 0:
            raise RuntimeError("No data read from encoded file")

        return encoded_data

    except subprocess.TimeoutExpired:
        raise RuntimeError("In-memory encoding timed out")
    except Exception as e:
        raise RuntimeError(f"In-memory encoding failed: {str(e)}")
    finally:
        
        for temp_file in [input_temp.name, output_temp.name]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except (OSError, PermissionError):
                pass 

def get_tier1_samples(input_path: str, log_callback, task_id: int, metrics: dict) -> Optional[Tuple[List[float], Dict[str, float]]]:
    """Tier 1: Uses keyframe detection to find sample points and analyze complexity."""
    try:
        cmd = [SETTINGS.ffmpeg_path, '-skip_frame', 'nokey', '-i', input_path, '-vf', 'showinfo', '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=FFMPEG_ENV, timeout=120)
        timestamps = []
        for line in result.stderr.split('\n'):
            if 'pts_time:' in line:
                time_m = re.search(r'pts_time:([\d\.]+)', line)
                if time_m:
                    timestamps.append(float(time_m.group(1)))

        
        video_duration = metrics['video_duration_seconds']
        timestamps = [
            ts for ts in timestamps
            if SETTINGS.skip_start_seconds <= ts <= (video_duration - SETTINGS.skip_end_seconds)
        ]
        

        if len(timestamps) < SETTINGS.min_keyframes_required:
            log_callback(task_id, f"Tier 1: Found only {len(timestamps)} keyframes in the allowed range, need {SETTINGS.min_keyframes_required}.")
            return None

        
        keyframe_intervals = []
        for i in range(1, len(timestamps)):
            keyframe_intervals.append(timestamps[i] - timestamps[i-1])

        avg_keyframe_interval = sum(keyframe_intervals) / len(keyframe_intervals) if keyframe_intervals else 0
        min_keyframe_interval = min(keyframe_intervals) if keyframe_intervals else 0
        keyframes_per_minute = (len(timestamps) / video_duration) * 60 if video_duration > 0 else 0

        
        
        
        complexity_score = min(1.0, 1.0 - (avg_keyframe_interval / 10.0))

        complexity_data = {
            'scene_count': len(timestamps), 
            'avg_scene_duration': avg_keyframe_interval,
            'min_scene_duration': min_keyframe_interval,
            'scenes_per_minute': keyframes_per_minute,
            'quick_cut_ratio': 0.0, 
            'complexity_score': complexity_score,
            'sampling_method': 'tier1'
        }

        log_callback(task_id, f"Tier 1: Found {len(timestamps)} keyframes, complexity score: {complexity_score:.2f}")

        indices = [int(i * (len(timestamps) - 1) / (SETTINGS.num_samples - 1)) for i in range(SETTINGS.num_samples)]
        selected = sorted([timestamps[i] for i in set(indices)])

        log_callback(task_id, f"Tier 1: Selected {len(selected)} samples based on keyframes.")
        return selected, complexity_data
    except Exception:
        return None

def get_tier2_samples(metrics: dict, log_callback, task_id: int) -> List[float]:
    """Tier 2: Gets sample points based on even intervals, with options to skip start/end."""
    video_duration = metrics['video_duration_seconds']
    use_full_duration = False

    effective_duration = video_duration - SETTINGS.skip_start_seconds - SETTINGS.skip_end_seconds
    required_time = SETTINGS.num_samples * SETTINGS.sample_segment_duration

    if effective_duration < required_time:
        log_callback(task_id, "[yellow]Tier 2: Not enough time for samples after skipping start/end. Using full video duration.[/yellow]")
        use_full_duration = True

    if use_full_duration:
        start_offset = 0
        duration_to_use = video_duration
    else:
        start_offset = SETTINGS.skip_start_seconds
        duration_to_use = effective_duration

    interval = (duration_to_use - required_time) / (SETTINGS.num_samples - 1) if SETTINGS.num_samples > 1 and duration_to_use > required_time else 0

    return [start_offset + (i * (SETTINGS.sample_segment_duration + interval)) for i in range(SETTINGS.num_samples)]

def get_final_sample_points(input_path: str, task_id: int, log_callback, metrics: dict) -> Tuple[List[float], Optional[Dict[str, float]]]:
    """Determines the best sample points using a tiered approach, returning complexity data if available."""
    samples = None
    complexity_data = None
    video_duration = metrics['video_duration_seconds']

    if SETTINGS.sampling_method == 'tier0':
        log_callback(task_id, "Attempting PySceneDetect - Tier 0...")
        if PYSCENEDETECT_AVAILABLE:
            result = get_tier0_samples(input_path, log_callback, task_id, video_duration)
            if result:
                samples, complexity_data = result
                return samples, complexity_data
            log_callback(task_id, "Tier 0 failed, falling back to Tier 1...")
        else:
            log_callback(task_id, "[yellow]Tier 0 selected, but PySceneDetect is not installed. Falling back.[/yellow]")

    if SETTINGS.sampling_method in ['tier0', 'tier1']:
        log_callback(task_id, "Attempting Keyframe Detection - Tier 1...")
        result = get_tier1_samples(input_path, log_callback, task_id, metrics)
        if result:
            samples, complexity_data = result
            return samples, complexity_data
        log_callback(task_id, "Tier 1 failed, falling back to Tier 2...")

    log_callback(task_id, "Using Time Intervals - Tier 2...")
    samples = get_tier2_samples(metrics, log_callback, task_id)

    
    complexity_data = {
        'scene_count': 0,
        'avg_scene_duration': 0,
        'min_scene_duration': 0,
        'scenes_per_minute': 0,
        'quick_cut_ratio': 0,
        'complexity_score': 0.5, 
        'sampling_method': 'tier2'
    }

    return samples, complexity_data



def create_master_sample_in_memory(input_path: str, sample_points: List[float], task_id: int, log_callback, video_stream: Dict) -> tuple[bytes, float]:
    """Creates master sample in memory by concatenating segments via pipes, preserving all color info."""
    start_time = time.time()
    log_callback(task_id, f"Creating master sample in RAM from {len(sample_points)} segments...")

    color_args = build_color_args(video_stream)
    pix_fmt_arg = ['-pix_fmt', video_stream['pix_fmt']] if video_stream.get('pix_fmt') else []

    
    filter_parts = []
    for i, start_s in enumerate(sample_points):
        filter_parts.append(f"[0:v]trim=start={start_s}:duration={SETTINGS.sample_segment_duration},setpts=PTS-STARTPTS[seg{i}]")

    concat_filter = "".join([f"[seg{i}]" for i in range(len(sample_points))]) + f"concat=n={len(sample_points)}:v=1:a=0[out]"
    filter_complex = ";".join(filter_parts) + ";" + concat_filter

    if SETTINGS.master_sample_encoder == 'nvenc':
        encoder_cmd = ['-c:v', 'h264_nvenc', '-preset', 'p1', '-cq', '0']
    elif SETTINGS.master_sample_encoder == 'raw':
        encoder_cmd = ['-c:v', 'rawvideo']
    else:
        encoder_cmd = ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0']

    cmd = [
        SETTINGS.ffmpeg_path, '-i', input_path,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        *pix_fmt_arg, *color_args, *encoder_cmd,
        '-f', 'matroska', 'pipe:1'
    ]

    result = subprocess.run(cmd, capture_output=True, env=FFMPEG_ENV, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Master sample creation failed: {result.stderr.decode('utf-8', errors='ignore')}")

    log_callback(task_id, "[green]Master sample created in RAM.[/green]")
    return result.stdout, time.time() - start_time

def find_best_cq(input_path: str, task_id: int, log_callback, metrics: dict, worker_progress_objects: dict, lock: threading.Lock) -> tuple[Optional[int], dict]:
    """Performs a binary search to find the optimal CQ/CRF value to meet the target VMAF."""
    timings = {}
    video_stream = next((s for s in metrics.get('media_info', {}).get('streams', []) if s.get('codec_type') == 'video'), None)
    if not video_stream:
        log_callback(task_id, "[bold red]Could not find video stream information for CQ search.[/bold red]")
        return None, {}

    try:
        sample_points_list, complexity_data = get_final_sample_points(input_path, task_id, log_callback, metrics)

        if complexity_data:
            metrics['complexity_data'] = complexity_data
            complexity_score = complexity_data.get('complexity_score', 0.5)

        master_sample_data, timings['sample_creation_time'] = create_master_sample_in_memory(
            input_path, sample_points_list, task_id, log_callback, video_stream
        )

        tested_cqs, low_cq, high_cq = {}, SETTINGS.cq_search_min, SETTINGS.cq_search_max
        total_vmaf_work_time, actual_vmaf_tests = 0.0, 0

        class IterationProgressTracker:
            def __init__(self):
                self.lock = threading.Lock()
                self.subtask_progress = {}
                self.completed_count = 0
                self.total_subtasks = 0
                self.vmaf_progress = None
                self.iter_task_id = None

            def initialize(self, vmaf_progress, iter_task_id, num_subtasks):
                with self.lock:
                    self.vmaf_progress = vmaf_progress
                    self.iter_task_id = iter_task_id
                    self.total_subtasks = num_subtasks
                    self.subtask_progress = {i: 0.0 for i in range(num_subtasks)}
                    self.completed_count = 0

            def update_subtask_progress(self, subtask_id: int, progress: float):
                with self.lock:
                    if subtask_id in self.subtask_progress:
                        old_progress = self.subtask_progress[subtask_id]
                        self.subtask_progress[subtask_id] = progress

                        if progress >= 100.0 and old_progress < 100.0:
                            self.completed_count += 1
                        
                        if self.total_subtasks > 0:
                            total_weighted_progress = sum(self.subtask_progress.values())
                            average_progress = total_weighted_progress / self.total_subtasks

                            if self.vmaf_progress and self.iter_task_id is not None:
                                with lock:
                                    self.vmaf_progress.update(self.iter_task_id, completed=average_progress)

        def run_single_vmaf_test(cq, progress_tracker=None, subtask_id=None):
            nonlocal actual_vmaf_tests, total_vmaf_work_time

            if SETTINGS.enable_cache:
                complexity_score = complexity_data.get('complexity_score', None) if complexity_data else None
                cached_score = vmaf_cache.get(input_path, sample_points_list, cq, complexity_score)
                if cached_score is not None:
                    log_callback(task_id, f"  -> Found in cache for CQ/CRF {cq}: {cached_score:.2f}")
                    if progress_tracker and subtask_id is not None:
                        progress_tracker.update_subtask_progress(subtask_id, 100.0)
                    tested_cqs[cq] = cached_score
                    total_vmaf_work_time += 0.1
                    return cq, cached_score

            test_start_time = time.time()
            actual_vmaf_tests += 1

            if cq in tested_cqs:
                if progress_tracker and subtask_id is not None:
                    progress_tracker.update_subtask_progress(subtask_id, 100.0)
                return cq, tested_cqs[cq]

            try:
                def vmaf_progress_callback(progress_pct):
                    if progress_tracker and subtask_id is not None:
                        progress_tracker.update_subtask_progress(subtask_id, progress_pct)
                
                encoded_sample_data = encode_sample_in_memory(master_sample_data, cq, video_stream)
                
                score = run_vmaf_comparison_in_memory(
                    encoded_sample_data,
                    master_sample_data,
                    vmaf_progress_callback
                )

                tested_cqs[cq] = score
                if SETTINGS.enable_cache:
                    complexity_score = complexity_data.get('complexity_score', None) if complexity_data else None
                    vmaf_cache.set(input_path, sample_points_list, cq, score, complexity_score)

                total_vmaf_work_time += time.time() - test_start_time
                vmaf_progress_callback(100.0)
                return cq, score

            except Exception as e:
                log_callback(task_id, f"[red]VMAF test failed for CQ {cq}: {e}[/red]")
                if progress_tracker and subtask_id is not None:
                    progress_tracker.update_subtask_progress(subtask_id, 100.0)
                return cq, 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=SETTINGS.num_parallel_vmaf_runs) as executor:
            vmaf_progress = Progress(
                TextColumn("[cyan]{task.description}[/cyan]"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeRemainingColumn()
            )

            with lock:
                worker_progress_objects[task_id] = {'vmaf_search': vmaf_progress}

            for iteration in range(SETTINGS.max_iterations):
                if high_cq - low_cq <= 1:
                    break

                # --- NEW LOGIC TO GUARANTEE THE CORRECT NUMBER OF TESTS ---
                test_cqs_for_iteration = []
                # Always add boundaries first if they haven't been tested yet.
                if low_cq not in tested_cqs:
                    test_cqs_for_iteration.append(low_cq)
                if high_cq not in tested_cqs and high_cq not in test_cqs_for_iteration:
                    test_cqs_for_iteration.append(high_cq)

                # Calculate how many "interior" points we can test to fill up the parallel runs.
                num_interior_points = SETTINGS.num_parallel_vmaf_runs - len(test_cqs_for_iteration)
                
                if num_interior_points > 0:
                    # Divide the remaining interval to place the interior points.
                    step = (high_cq - low_cq) / (num_interior_points + 1)
                    for j in range(num_interior_points):
                        point = round(low_cq + (j + 1) * step)
                        # Ensure the point is actually inside the boundaries and unique.
                        if point > low_cq and point < high_cq and point not in test_cqs_for_iteration:
                            test_cqs_for_iteration.append(point)

                # Final cleanup to ensure uniqueness and order.
                test_cqs_for_iteration = sorted(list(set(test_cqs_for_iteration)))
                
                # Filter out any points that have already been tested in previous iterations.
                test_cqs_for_iteration = [cq for cq in test_cqs_for_iteration if cq not in tested_cqs]

                # --- END OF NEW LOGIC ---

                if not test_cqs_for_iteration:
                    if high_cq - low_cq > 1:
                        # If all calculated points were already tested, pick one in the middle as a fallback.
                        mid_point = round((low_cq + high_cq) / 2)
                        if mid_point not in tested_cqs:
                             test_cqs_for_iteration = [mid_point]
                        else:
                            break # The area is fully covered.
                    else:
                        break # Exit if the range is too small to test.

                log_callback(task_id, f"Iteration {iteration + 1}/{SETTINGS.max_iterations}: Testing CQ/CRFs {test_cqs_for_iteration}")

                progress_tracker = IterationProgressTracker()
                iter_task_id = vmaf_progress.add_task(f"VMAF Iteration {iteration + 1}", total=100)
                progress_tracker.initialize(vmaf_progress, iter_task_id, len(test_cqs_for_iteration))

                futures = {}
                for i, cq in enumerate(test_cqs_for_iteration):
                    future = executor.submit(run_single_vmaf_test, cq, progress_tracker, i)
                    futures[future] = (i, cq)

                results = []
                for future in concurrent.futures.as_completed(futures):
                    subtask_id, cq = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        log_callback(task_id, f"[red]Future failed for CQ {cq}: {e}[/red]")
                        results.append((cq, 0.0))
                
                with lock:
                    vmaf_progress.update(iter_task_id, completed=100.0)

                vmaf_progress.remove_task(iter_task_id)
                results = sorted(results)

                color_palette = ["cyan", "magenta", "yellow", "green", "blue", "bright_red"]
                results_table = Table.grid(padding=(0, 2))
                for _ in range(len(results)):
                    results_table.add_column()

                cq_row = [
                    Text(f"CQ {cq}", style=color_palette[i % len(color_palette)], justify="center")
                    for i, (cq, _) in enumerate(results)
                ]
                score_row = [
                    Text(f"{score:.2f}", style=color_palette[i % len(color_palette)], justify="center")
                    for i, (_, score) in enumerate(results)
                ]

                results_table.add_row(*cq_row)
                results_table.add_row(*score_row)
                log_callback(task_id, results_table)

                passing_cqs = {cq for cq, score in tested_cqs.items()
                               if score >= (SETTINGS.target_vmaf - SETTINGS.vmaf_tolerance)}
                failing_cqs = {cq for cq, score in tested_cqs.items()
                               if score < (SETTINGS.target_vmaf - SETTINGS.vmaf_tolerance)}

                if passing_cqs:
                    low_cq = max(passing_cqs)
                if failing_cqs:
                    high_cq = min(failing_cqs)
        
        if actual_vmaf_tests > 0:
            timings['vmaf_search_time'] = total_vmaf_work_time
            log_callback(task_id, f"[dim]VMAF search took {total_vmaf_work_time:.1f}s (workload) with {actual_vmaf_tests} actual tests.[/dim]")
        else:
            timings['vmaf_search_time'] = 0.0
            log_callback(task_id, f"[dim]VMAF search was fully cached.[/dim]")
        
        final_passing_cqs = {cq for cq, score in tested_cqs.items()
                             if score >= (SETTINGS.target_vmaf - SETTINGS.vmaf_tolerance)}

        if final_passing_cqs:
            final_cq = max(final_passing_cqs)
        elif tested_cqs:
            final_cq = max(tested_cqs, key=tested_cqs.get)
        else:
            log_callback(task_id, "[bold red]VMAF search yielded no results. Using fallback.[/bold red]")
            return SETTINGS.cq_search_max, timings


        best_vmaf = tested_cqs.get(final_cq, 0) if tested_cqs else 0
        target_min = SETTINGS.target_vmaf - SETTINGS.vmaf_tolerance
        target_achieved = best_vmaf >= target_min

        if target_achieved:
            log_callback(task_id, f"[bold green]✅ Target VMAF {SETTINGS.target_vmaf} achieved with CQ/CRF {final_cq} (VMAF: {best_vmaf:.1f})[/bold green]")
            log_callback(task_id, f"[green]Proceeding with final encoding...[/green]")
        else:
            shortfall = target_min - best_vmaf
            log_callback(task_id, f"[yellow]⚠️  Target VMAF {SETTINGS.target_vmaf} not achievable within CQ/CRF range {SETTINGS.cq_search_min}-{SETTINGS.cq_search_max}[/yellow]")
            log_callback(task_id, f"[yellow]Best achievable: CQ/CRF {final_cq} (VMAF: {best_vmaf:.1f}, shortfall: {shortfall:.1f} points)[/yellow]")
            
            if SETTINGS.skip_encoding_if_target_not_reached:
                log_callback(task_id, f"[yellow]Skipping encoding as per configuration settings.[/yellow]")
                return None, timings
            else:
                log_callback(task_id, f"[yellow]Proceeding with best available quality...[/yellow]")

        return final_cq, timings

    except Exception as e:
        log_callback(task_id, f"[bold red]FATAL error in CQ search: {type(e).__name__}: {str(e)}[/bold red]")
        import traceback
        log_callback(task_id, f"[dim]{traceback.format_exc()}[/dim]")
        return SETTINGS.cq_search_max, timings
    finally:
        with lock:
            if task_id in worker_progress_objects and 'vmaf_search' in worker_progress_objects[task_id]:
                worker_progress_objects[task_id].pop('vmaf_search', None)

def log_enhanced_results(input_path: str, output_path: str, cq_used: int, duration: float, status: str, metrics: Dict, timings: dict):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    input_size = metrics.get('input_size_bytes', 0)
    video_duration = metrics.get('video_duration_seconds', 0)
    
    output_size = Path(output_path).stat().st_size if "Success" in status and Path(output_path).exists() else 0
    source_bitrate = (input_size * 8) / (video_duration * 1000) if video_duration > 0 else 0
    output_bitrate = (output_size * 8) / (video_duration * 1000) if video_duration > 0 and output_size > 0 else 0
    space_saved = (input_size - output_size) / (1024 * 1024) if output_size > 0 else 0
    compression_ratio = input_size / output_size if output_size > 0 else 0
    processing_speed = video_duration / duration if duration > 0 else 0
    memory_mb = memory_manager.get_usage_mb()
    log_entry = (f"--- Encoding Log ({timestamp}) ---\n"
                 f"File: {Path(input_path).name}\nStatus: {status}\n"
                 f"Encoder: {SETTINGS.encoder_type}\nCQ/CRF Value: {cq_used}\n"
                 f"Input Size: {input_size / (1024*1024):.2f} MB\nOutput Size: {output_size / (1024*1024):.2f} MB\n"
                 f"Space Saved: {space_saved:.2f} MB\nCompression Ratio: {compression_ratio:.2f}x\n"
                 f"Source Bitrate: {source_bitrate:.2f} kb/s\nOutput Bitrate: {output_bitrate:.2f} kb/s\n"
                 f"Processing Time: {format_duration(duration)}\nProcessing Speed: {processing_speed:.2f}x\n"
                 f"Peak Memory Usage: {memory_mb:.1f} MB\n\n")
    try:
        with open(SETTINGS.database_path, 'a', encoding='utf-8') as f: f.write(log_entry)
    except IOError as e: console.print(f"[red]Failed to write log: {e}[/red]")

    if SETTINGS.enable_performance_cache and "Success" in status:
        video_stream = next((s for s in metrics.get('media_info', {}).get('streams', []) if s.get('codec_type') == 'video'), None)
        if video_stream:
            framerate_str = video_stream.get('avg_frame_rate', '30/1')
            num, den = framerate_str.split('/') if '/' in framerate_str else (framerate_str, 1)
            framerate = float(num) / float(den) if float(den) > 0 else 30
            total_frames = video_duration * framerate
            timings['final_encode_fps'] = total_frames / duration if duration > 0 else 0
            performance_db.log_performance(metrics, timings)

def rename_skipped_file(filepath: str, task_id: int, log_callback):
    """Renames a file with the specified suffix if the setting is enabled."""
    if not SETTINGS.rename_skipped_files:
        return

    try:
        p_input = Path(filepath)
        if not p_input.exists():
            return

        new_name = f"{p_input.stem}{SETTINGS.skipped_file_suffix}{p_input.suffix}"
        new_path = p_input.parent / new_name

        if new_path.exists():
            log_callback(task_id, f"[yellow]Skipping rename: '{new_path.name}' already exists.[/yellow]")
            return

        p_input.rename(new_path)
        log_callback(task_id, f"[dim]Renamed source to '{new_path.name}'[/dim]")

    except Exception as e:
        log_callback(task_id, f"[red]Error renaming file {filepath}: {e}[/red]")


def read_stderr_non_blocking(process, timeout=30):

    if os.name == 'nt':  # Windows-specific implementation
        q = queue.Queue()

        def enqueue_output(out, q):
            try:
                for line in iter(out.readline, ''):
                    q.put(line)
            finally:
                out.close()
                q.put(None)  # Signal that we are done

        t = threading.Thread(target=enqueue_output, args=(process.stderr, q), daemon=True)
        t.start()

        while process.poll() is None or not q.empty():
            try:
                line = q.get(timeout=timeout)
                if line is None:  # End-of-stream signal
                    break
                yield line
            except queue.Empty:
                yield None  # Timeout signal
    else:  # Unix-like implementation
        while True:
            ready, _, _ = select.select([process.stderr], [], [], timeout)
            if not ready:
                if process.poll() is not None:
                    break
                yield None  # Timeout signal
                continue

            line = process.stderr.readline()
            if not line:  # End-of-stream
                break
            yield line

def run_encode(input_path: str, cq_value: int, task_id: int, batch_state: Dict, lock: threading.Lock, metrics: dict, log_callback, worker_progress, worker_task_id, timings: dict) -> bool:
    """Runs the final encode with robust monitoring and completion detection."""
    start_time = time.time()
    p_input = Path(input_path)

    output_dir = Path(SETTINGS.output_directory) if SETTINGS.output_directory and SETTINGS.output_directory.strip() else p_input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_output = output_dir / f"{p_input.stem}_temp{p_input.suffix}"
    final_output = output_dir / f"{p_input.stem}{SETTINGS.output_suffix}{p_input.suffix}"

    # --- ADDED: Cleanup for any previous failed runs ---
    if temp_output.exists():
        try:
            temp_output.unlink()
        except OSError:
            pass # Ignore if we can't delete it right away

    video_stream = next((s for s in metrics.get('media_info', {}).get('streams', []) if s.get('codec_type') == 'video'), None)
    if not video_stream:
        log_callback(task_id, "[red]Could not find video stream. Cannot proceed with final encode.[/red]")
        return False

    encoder_args = build_encoder_args(cq_value, video_stream, for_final_encode=True)
    cmd = [
        SETTINGS.ffmpeg_path, '-y', '-i', str(p_input),
        *encoder_args,
        '-map', '0:v', '-map', '0:a?', '-map', '0:s?',
        '-c:a', 'copy', '-c:s', 'copy',
        '-f', 'matroska', str(temp_output)
    ]

    log_callback(task_id, f"Starting final encode with {SETTINGS.encoder_type.upper()} (CQ/CRF: {cq_value})...")
    
    # --- ADDED: All the monitoring state variables ---
    last_size = 0
    last_activity_time = time.time()
    last_progress_time = time.time()
    last_reported_progress = 0
    stuck_at_100_time = None
    size_at_100 = None
    monitor_exception = None

   
    MAX_IDLE_TIME = 300  # 5 minutes with no file growth
    MAX_STUCK_AT_100_TIME = 120  # 2 minutes stuck at 100%
    PROGRESS_TIMEOUT = 600  # 10 minutes with no progress updates from stderr

    try:
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1, encoding='utf-8', env=FFMPEG_ENV)
        duration = metrics['video_duration_seconds']
        time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
        
 
        stop_monitor = threading.Event()
        
        def monitor_file_activity():
            nonlocal last_size, last_activity_time, monitor_exception
            try:
                while not stop_monitor.is_set():
                    if temp_output.exists():
                        try:
                            current_size = temp_output.stat().st_size
                            if current_size > last_size:
                                last_size = current_size
                                last_activity_time = time.time()
                        except (OSError, IOError):
                            pass # File might be locked, that's okay

                    # Check if we've been idle too long
                    idle_time = time.time() - last_activity_time
                    if idle_time > MAX_IDLE_TIME:
                        monitor_exception = RuntimeError(f"File hasn't grown for {idle_time:.0f}s (stalled)")
                        break
                    
                    stop_monitor.wait(5)  # Check every 5 seconds
            except Exception as e:
                monitor_exception = e
        
        monitor_thread = threading.Thread(target=monitor_file_activity, daemon=True)
        monitor_thread.start()
        
   
        for line in read_stderr_non_blocking(process, timeout=30):
            if line is None:  # Timeout occurred
                if process.poll() is not None: break # Process exited, stop reading
                if time.time() - last_progress_time > PROGRESS_TIMEOUT:
                    raise RuntimeError(f"No progress updates for {PROGRESS_TIMEOUT}s")
                continue # No new line, just continue waiting

            # We have a line from FFmpeg, so reset the progress timer
            last_progress_time = time.time()
            m = time_pattern.search(line)
            if m and duration > 0:
                h, m_str, s, ms = map(int, m.groups())
                current_time = h * 3600 + m_str * 60 + s + ms/100
                progress = min(100.0, (current_time / duration) * 100)
                
                if progress >= 99.9:
                    if stuck_at_100_time is None:
                        stuck_at_100_time = time.time()
                        size_at_100 = last_size
                        log_callback(task_id, "[yellow]Reached 100%, waiting for finalization...[/yellow]")
                
                if progress > last_reported_progress:
                    with lock: worker_progress.update(worker_task_id, completed=progress)
                    last_reported_progress = progress
            
            # Check if the background monitor found a problem
            if monitor_exception:
                raise monitor_exception

     
        stop_monitor.set()
        monitor_thread.join(timeout=5)
        
        log_callback(task_id, "[dim]Progress parsing finished. Waiting for FFmpeg process to exit...[/dim]")
        try:
            process.wait(timeout=60) # Give FFmpeg 1 minute to close gracefully
        except subprocess.TimeoutExpired:
            log_callback(task_id, "[yellow]FFmpeg did not exit cleanly. Terminating...[/yellow]")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log_callback(task_id, "[red]FFmpeg did not respond to terminate. Killing...[/red]")
                process.kill()
                process.wait(timeout=10)
        
        if process.returncode != 0 and process.returncode is not None:
             log_callback(task_id, f"[red]Encoding failed: FFmpeg returned code {process.returncode}[/red]")
             return False
        
        if not temp_output.exists() or temp_output.stat().st_size < 1024:
            log_callback(task_id, "[red]Encoding produced an empty or missing file.[/red]")
            return False

        # --- ADDED: Final verification with ffprobe ---
        log_callback(task_id, "[dim]Verifying output file integrity...[/dim]")
        output_info = get_media_info(str(temp_output))
        if not (output_info and 'format' in output_info):
            log_callback(task_id, "[red]Verification failed: Could not read info from encoded file.[/red]")
            return False
            
        output_duration = float(output_info['format'].get('duration', 0))
        duration_diff = abs(output_duration - metrics['video_duration_seconds'])
        if duration_diff > 2.0: # Allow 2 second difference
            log_callback(task_id, f"[red]Verification failed: Duration mismatch! Expected ~{metrics['video_duration_seconds']:.1f}s, Got {output_duration:.1f}s.[/red]")
            return False

        # --- MODIFIED: Rest of the logic remains similar but is now more robust ---
        output_size, input_size = temp_output.stat().st_size, metrics['input_size_bytes']
        size_reduction_percent = ((input_size - output_size) / input_size) * 100

        if size_reduction_percent < SETTINGS.min_size_reduction_threshold:
            log_callback(task_id, f"[yellow]Insufficient size reduction ({size_reduction_percent:.1f}% < {SETTINGS.min_size_reduction_threshold}%), keeping original file.[/yellow]")
            rename_skipped_file(input_path, task_id, log_callback)
            return False

        if final_output.exists(): final_output.unlink()
        temp_output.rename(final_output)

        if SETTINGS.delete_source_file:
            p_input.unlink()
            log_callback(task_id, f"[green]Success: Source file deleted, encoded file saved (reduced by {size_reduction_percent:.1f}%).[/green]")
        else:
            log_callback(task_id, f"[green]Success: Source file kept, encoded file saved (reduced by {size_reduction_percent:.1f}%).[/green]")
        
        with lock:
            worker_progress.update(worker_task_id, completed=100, description="Encoded")
            batch_state['deleted_source_size'] += input_size
            batch_state['encoded_output_size'] += output_size
        
        log_enhanced_results(str(p_input), str(final_output), cq_value, time.time() - start_time, "Success", metrics, timings)
        return True

    except Exception as e:
        log_callback(task_id, f"[red]Encoding error: {e}[/red]")
        # Ensure process is terminated on any error
        if 'process' in locals() and process.poll() is None:
            process.kill()
            process.wait()
        return False
    finally:
        # Final cleanup of temp file
        if temp_output.exists() and not final_output.exists():
            try:
                temp_output.unlink()
            except OSError:
                pass

def handle_failed_file(filepath: str, task_id: int, lock: threading.Lock, batch_state: Dict, reason: str, metrics: dict):
    """Handles logging and state updates for files that are skipped or fail."""
    log_enhanced_results(filepath, "", 0, 0, f"Skipped: {reason}", metrics, {})
    with lock: not_replaced_files.append(Path(filepath).name)
    rename_skipped_file(filepath, task_id, log_to_worker)


def enhanced_task_worker(filepath: str, task_id: int, lock: threading.Lock, batch_state: Dict, worker_progress_objects: Dict):
    """The main worker function that processes a single video file from start to finish."""
    metrics = {}
    task_start_time = time.time()
    try:
        log_to_worker(task_id, f"Processing: {Path(filepath).name}")
        media_info = get_media_info(filepath)
        if not (media_info and 'format' in media_info and float(media_info.get('format', {}).get('duration', 0)) > 0):
            return handle_failed_file(filepath, task_id, lock, batch_state, "Media info failed", {'input_size_bytes': get_file_size_info(filepath)[0]})

        file_size, size_mb = get_file_size_info(filepath)
        metrics = {'input_size_bytes': file_size, 'input_size_mb': size_mb, 'video_duration_seconds': float(media_info['format']['duration']), 'media_info': media_info}

        
        if SETTINGS.min_duration_seconds > 0 and metrics['video_duration_seconds'] < SETTINGS.min_duration_seconds:
            log_to_worker(task_id, f"[yellow]Skipping: duration < {SETTINGS.min_duration_seconds}s.[/yellow]"); return handle_failed_file(filepath, task_id, lock, batch_state, "Below duration threshold", metrics)
        if SETTINGS.min_filesize_mb > 0 and metrics['input_size_mb'] < SETTINGS.min_filesize_mb:
            log_to_worker(task_id, f"[yellow]Skipping: filesize < {SETTINGS.min_filesize_mb}MB.[/yellow]"); return handle_failed_file(filepath, task_id, lock, batch_state, "Below filesize threshold", metrics)

        video_stream = next((s for s in media_info.get('streams', []) if s.get('codec_type') == 'video'), None)
        if video_stream:
            width, source_bitrate_kbps = video_stream.get('width', 0), (metrics['input_size_bytes'] * 8) / (metrics['video_duration_seconds'] * 1000)
            if width >= 3840 and SETTINGS.min_bitrate_4k_kbps > 0 and source_bitrate_kbps < SETTINGS.min_bitrate_4k_kbps:
                log_to_worker(task_id, f"[yellow]Skipping 4K: bitrate < {SETTINGS.min_bitrate_4k_kbps}kbps.[/yellow]"); return handle_failed_file(filepath, task_id, lock, batch_state, "Below 4K bitrate", metrics)
            elif 1920 <= width < 3840 and SETTINGS.min_bitrate_1080p_kbps > 0 and source_bitrate_kbps < SETTINGS.min_bitrate_1080p_kbps:
                log_to_worker(task_id, f"[yellow]Skipping 1080p: bitrate < {SETTINGS.min_bitrate_1080p_kbps}kbps.[/yellow]"); return handle_failed_file(filepath, task_id, lock, batch_state, "Below 1080p bitrate", metrics)
            elif 1280 <= width < 1920 and SETTINGS.min_bitrate_720p_kbps > 0 and source_bitrate_kbps < SETTINGS.min_bitrate_720p_kbps:
                log_to_worker(task_id, f"[yellow]Skipping 720p: bitrate < {SETTINGS.min_bitrate_720p_kbps}kbps.[/yellow]"); return handle_failed_file(filepath, task_id, lock, batch_state, "Below 720p bitrate", metrics)

        optimal_cq, timings = find_best_cq(filepath, task_id, log_to_worker, metrics, worker_progress_objects, lock)
        if optimal_cq is None: 
            failure_reason = "Target quality not achievable" if SETTINGS.skip_encoding_if_target_not_reached else "CQ optimization failed"
            return handle_failed_file(filepath, task_id, lock, batch_state, failure_reason, metrics)

        encode_progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"))
        encode_task_id = encode_progress.add_task("Encoding...", total=100)
        with lock: worker_progress_objects[task_id] = {'encode': encode_progress}

        success = run_encode(filepath, optimal_cq, task_id, batch_state, lock, metrics, log_to_worker, encode_progress, encode_task_id, timings)
        if not success:
            with lock: not_replaced_files.append(Path(filepath).name)

    except Exception as e:
        log_to_worker(task_id, f"[bold red]Unexpected worker error: {e}[/bold red]"); import traceback; log_to_worker(task_id, f"[dim]{traceback.format_exc()}[/dim]")
        handle_failed_file(filepath, task_id, lock, batch_state, f"Exception: {e}", metrics)
    finally:
        actual_time_taken = time.time() - task_start_time
        with lock:
            batch_state['files_completed'] += 1
            processed_size = metrics.get('input_size_bytes', 0)
            batch_state['processed_size'] += processed_size


            if processed_size > 0 and SETTINGS.enable_performance_cache:
                
                complexity_data = metrics.get('complexity_data')
                if complexity_data:
                    predicted_time = performance_db.get_complexity_adjusted_eta(metrics, complexity_data)
                else:
                    predicted_time = performance_db.predict_eta(metrics)

                batch_state['predicted_completed_time'] += predicted_time
                batch_state['predicted_remaining_time'] -= predicted_time
                batch_state['actual_completed_time'] += actual_time_taken

                if batch_state['predicted_completed_time'] > 0:
                    ratio = batch_state['actual_completed_time'] / batch_state['predicted_completed_time']
                    old_ratio = batch_state.get('current_performance_ratio', 1.0)
                    batch_state['current_performance_ratio'] = (old_ratio * 0.7) + (ratio * 0.3)
                    batch_state['last_eta_update_time'] = time.time()


def setup_console_size():
    """Set up optimal console size for the encoding script."""
    try:
        if os.name == 'nt': 
            os.system('mode con: cols=120 lines=35')
    except:
        pass 

def get_current_eta(batch_state: Dict) -> float:
    """Lightning-fast ETA calculation using pre-calculated totals from the batch_state."""
    if batch_state['files_completed'] == 0:
        
        return batch_state['predicted_remaining_time']



    
    performance_ratio = batch_state.get('current_performance_ratio', 1.0)
    eta = batch_state['predicted_remaining_time'] * performance_ratio
    return eta

def get_countdown_eta(batch_state: Dict) -> float:
    """
    Calculates a real-time countdown ETA.
    It takes the last smart calculation and subtracts time elapsed since then.
    """
    
    smart_eta = get_current_eta(batch_state)

    
    time_since_update = time.time() - batch_state.get('last_eta_update_time', time.time())

    
    countdown_eta = smart_eta - time_since_update

    return max(0, countdown_eta)

def main():
    """The main entry point of the script."""
    setup_console_size()
    if not load_settings():
        input("Press Enter to exit..."); return

    global FFMPEG_ENV, memory_manager, vmaf_cache, performance_db
    FFMPEG_ENV = get_ffmpeg_env()
    memory_manager = MemoryManager(SETTINGS.memory_limit_gb)
    vmaf_cache = VMAFCache(SETTINGS.cache_database_path)
    performance_db = PerformanceDB(SETTINGS.performance_database_path)

    if not test_enhanced_setup():
        input("Press Enter to exit..."); return

    
    if SETTINGS.use_different_input_directory and SETTINGS.input_directory.strip():
        base_dir = Path(SETTINGS.input_directory)
        if not base_dir.exists():
            console.print(f"[red]Input directory not found: {SETTINGS.input_directory}[/red]")
            input("Press Enter to exit..."); return
    else:
        base_dir = Path(__file__).parent if '__file__' in locals() else Path('.')

    video_files = [str(p) for p in base_dir.rglob('*') if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".ts", ".m2ts", ".mpg", ".mpeg", ".m4v", ".y4m"} and SETTINGS.output_suffix not in p.stem.lower() and SETTINGS.skipped_file_suffix not in p.stem.lower()]

    if not video_files:
        console.print("[red]No video files found to process.[/red]")
        input("Press Enter to exit..."); return

    
    console.print("[bold cyan]🎬 AUTO VMAF ENCODER 🚀 - Data-Driven Video Encoding[/bold cyan]", justify="center")
    console.print()


    display_compact_summary(SETTINGS, console)
    console.print()

    console.print(f"Found [green]{len(video_files)}[/green] video file(s) to process.", highlight=False)
    console.print("Tip: For an optimal console, avoid shrinking the window. Only enlarge vertically if needed.")


    total_queue_size = sum(get_file_size_info(f)[0] for f in video_files)

    batch_state = {
        'total_files': len(video_files), 
        'files_completed': 0, 
        'deleted_source_size': 0, 
        'encoded_output_size': 0, 
        'processed_size': 0, 
        'start_time': time.time(), 
        'last_eta_update_time': time.time(),
        'total_queue_size_bytes': total_queue_size
    }

    initial_total_prediction = 0
    file_metrics_map = {} 
    if SETTINGS.enable_performance_cache:
        with Progress(SpinnerColumn()) as progress:
            task = progress.add_task("Scanning", total=len(video_files))
            for f in video_files:
                info = get_media_info(f)
                if info and 'format' in info and 'streams' in info:
                    metrics = {'video_duration_seconds': float(info['format']['duration']), 'media_info': info, 'input_size_bytes': get_file_size_info(f)[0]}
                    file_metrics_map[f] = metrics

                    
                    
                    video_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
                    if video_stream:
                        
                        bitrate = float(info['format'].get('bit_rate', 0))
                        resolution = classify_resolution(video_stream.get('width'), video_stream.get('height'))

                        
                        expected_bitrates = {'4K': 25000000, '1080p': 8000000, '720p': 5000000, 'SD': 2000000}
                        expected_bitrate = expected_bitrates.get(resolution, 8000000)

                        if bitrate > 0 and expected_bitrate > 0:
                            
                            bitrate_ratio = bitrate / expected_bitrate
                            estimated_complexity = min(1.0, max(0.0, (bitrate_ratio - 0.5) / 1.5))
                        else:
                            estimated_complexity = 0.5 

                        
                        estimated_complexity_data = {
                            'complexity_score': estimated_complexity,
                            'sampling_method': 'estimated'
                        }

                        initial_total_prediction += performance_db.get_complexity_adjusted_eta(metrics, estimated_complexity_data)
                    else:
                        initial_total_prediction += performance_db.predict_eta(metrics)
                else:
                    initial_total_prediction += 300 

                progress.update(task, advance=1)

    batch_state['predicted_remaining_time'] = initial_total_prediction
    batch_state['predicted_completed_time'] = 0.0
    batch_state['actual_completed_time'] = 0.0
    batch_state['current_performance_ratio'] = 1.0

    lock, task_id_generator, worker_progress_objects = threading.Lock(), itertools.count(1), {}

    stop_requested = False
    def wait_for_stop(): nonlocal stop_requested; input(); stop_requested = True
    threading.Thread(target=wait_for_stop, daemon=True).start()

    total_size = sum(get_file_size_info(f)[0] for f in video_files)
    total_progress = Progress(TextColumn("[bold cyan]Total Progress[/bold cyan]"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"))
    total_task_id = total_progress.add_task("Total", total=total_size)

    task_queue = video_files.copy()

    final_active_threads = []
    with Live(generate_layout(batch_state, total_progress, [], worker_progress_objects), console=console, refresh_per_second=5, vertical_overflow="ellipsis", transient=False) as live:
        with concurrent.futures.ThreadPoolExecutor(max_workers=SETTINGS.max_workers) as executor:
            futures = {}
            while True:
                if not stop_requested and len(futures) < SETTINGS.max_workers and task_queue:
                    filepath = task_queue.pop(0)
                    task_id = next(task_id_generator)
                    future = executor.submit(enhanced_task_worker, filepath, task_id, lock, batch_state, worker_progress_objects)
                    futures[future] = (task_id, filepath)

                done_futures = {f for f in futures if f.done()}
                for future in done_futures:
                    task_id, _ = futures.pop(future)
                    try: future.result()
                    except Exception as exc: log_to_worker(task_id, f"[bold red]Worker thread error: {exc}[/bold red]")

                with lock:
                    active_threads = [(tid, None, fname) for _, (tid, fname) in futures.items()]
                    if active_threads: final_active_threads = active_threads
                    total_progress.update(total_task_id, completed=batch_state['processed_size'])
                    live.update(generate_layout(batch_state, total_progress, active_threads, worker_progress_objects))

                if batch_state['files_completed'] == batch_state['total_files'] or (stop_requested and not futures):
                    break
                time.sleep(0.1)

        with lock:
            live.update(generate_layout(batch_state, total_progress, final_active_threads[-2:], worker_progress_objects))

    console.print(f"\n[green]✅ Encoding process completed![/]")
    if not_replaced_files:
        console.print("[yellow]The following files were not replaced (due to failure, insufficient size reduction, or being skipped):[/]")
        for f in not_replaced_files: console.print(f"- {f}")

    input("\nPress Enter to exit...")

if __name__ == '__main__':
    main()
