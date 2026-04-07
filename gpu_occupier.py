#!/usr/bin/env python3
"""
GPU Memory Occupier Script
Monitors all GPUs and occupies memory on any idle GPU found.
"""

import subprocess
import time
import argparse
import torch
import sys
from typing import List, Tuple

def get_gpu_info() -> List[dict]:
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 4:
                gpus.append({
                    'index': int(parts[0]),
                    'memory_total': int(parts[1]),
                    'memory_used': int(parts[2]),
                    'memory_free': int(parts[3])
                })
        return gpus
    except subprocess.CalledProcessError as e:
        print(f"Error getting GPU info: {e}")
        return []
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
        return []

def is_gpu_idle(gpu_info: dict, idle_threshold: float = 0.95) -> bool:
    """Check if GPU is idle (free memory > threshold of total memory)"""
    if gpu_info['memory_total'] == 0:
        return False
    free_ratio = gpu_info['memory_free'] / gpu_info['memory_total']
    return free_ratio >= idle_threshold

def occupy_gpu_memory(gpu_index: int, memory_mb: int, duration: int = None) -> torch.Tensor:
    """
    Occupy GPU memory by allocating tensors
    
    Args:
        gpu_index: GPU index to occupy
        memory_mb: Memory to occupy in MB
        duration: How long to hold memory (None = infinite)
    
    Returns:
        The allocated tensor (keep reference to prevent garbage collection)
    """
    device = torch.device(f'cuda:{gpu_index}')
    
    # Calculate number of elements needed (float32 = 4 bytes per element)
    num_elements = (memory_mb * 1024 * 1024) // 4
    
    # Allocate tensor on GPU
    tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
    
    # Touch the tensor to ensure memory is actually allocated
    tensor[0] = 1
    tensor[0] = 0
    
    print(f"✓ Occupied {memory_mb}MB on GPU {gpu_index}")
    
    if duration:
        print(f"  Will hold memory for {duration} seconds...")
        time.sleep(duration)
        del tensor
        torch.cuda.empty_cache()
        print(f"  Released memory on GPU {gpu_index}")
        return None
    
    return tensor

def monitor_and_occupy(
    occupancy_ratio: float = 0.8,
    check_interval: int = 5,
    max_duration: int = None,
    min_free_memory: int = 1024
):
    """
    Continuously monitor GPUs and occupy memory on idle ones
    
    Args:
        occupancy_ratio: Ratio of free memory to occupy (0.0-1.0)
        check_interval: Seconds between checks
        max_duration: Max time to hold memory per GPU (None = infinite)
        min_free_memory: Minimum free memory (MB) required to consider GPU idle
    """
    print("🔍 Starting GPU monitoring...")
    print(f"   Check interval: {check_interval}s")
    print(f"   Occupancy ratio: {occupancy_ratio*100:.0f}%")
    print(f"   Min free memory: {min_free_memory}MB")
    if max_duration:
        print(f"   Max duration: {max_duration}s")
    print("-" * 50)
    
    # Keep references to occupied tensors
    occupied_tensors = {}
    
    try:
        while True:
            gpus = get_gpu_info()
            
            if not gpus:
                print("No GPUs found!")
                time.sleep(check_interval)
                continue
            
            for gpu in gpus:
                gpu_idx = gpu['index']
                
                # Skip if already occupying this GPU
                if gpu_idx in occupied_tensors:
                    continue
                
                # Check if GPU is idle
                if is_gpu_idle(gpu) and gpu['memory_free'] >= min_free_memory:
                    # Calculate memory to occupy
                    memory_to_occupy = int(gpu['memory_free'] * occupancy_ratio)
                    
                    if memory_to_occupy >= min_free_memory:
                        print(f"\n⚡ Found idle GPU {gpu_idx}:")
                        print(f"   Total: {gpu['memory_total']}MB, Free: {gpu['memory_free']}MB")
                        print(f"   Occupying: {memory_to_occupy}MB")
                        
                        # Occupy memory
                        tensor = occupy_gpu_memory(gpu_idx, memory_to_occupy, max_duration)
                        
                        if max_duration:
                            # If temporary, remove from tracking after release
                            time.sleep(0.1)  # Small delay to allow release
                        else:
                            # Keep reference to prevent garbage collection
                            occupied_tensors[gpu_idx] = tensor
            
            # Print status
            active_gpus = len(occupied_tensors)
            total_gpus = len(gpus)
            print(f"\r📊 Status: Occupying {active_gpus}/{total_gpus} GPUs", end='', flush=True)
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
        print("Releasing all occupied memory...")
        for gpu_idx, tensor in occupied_tensors.items():
            del tensor
        torch.cuda.empty_cache()
        print("✅ All memory released!")

def main():
    parser = argparse.ArgumentParser(
        description='Monitor and occupy idle GPU memory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpu_occupier.py                    Default: occupy 80% of idle GPUs indefinitely
  python gpu_occupier.py -r 0.5 -i 10      Occupy 50% every 10 seconds
  python gpu_occupier.py -r 0.9 -d 3600    Occupy 90% for 1 hour
  python gpu_occupier.py -m 2048           Only occupy if more than 2GB free
        """
    )
    
    parser.add_argument('-r', '--ratio', type=float, default=0.8,
                       help='Ratio of free memory to occupy (0.0-1.0, default: 0.8)')
    parser.add_argument('-i', '--interval', type=int, default=5,
                       help='Check interval in seconds (default: 5)')
    parser.add_argument('-d', '--duration', type=int, default=None,
                       help='Max duration to hold memory in seconds (default: infinite)')
    parser.add_argument('-m', '--min-memory', type=int, default=1024,
                       help='Minimum free memory (MB) to consider GPU idle (default: 1024)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit instead of continuous monitoring')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.ratio <= 1:
        print("Error: Ratio must be between 0 and 1")
        sys.exit(1)
    
    if args.interval < 1:
        print("Error: Interval must be at least 1 second")
        sys.exit(1)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Please install PyTorch with CUDA support.")
        sys.exit(1)
    
    print(f"🚀 GPU Occupier Started")
    print(f"   Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print("-" * 50)
    
    if args.once:
        # Run once
        gpus = get_gpu_info()
        for gpu in gpus:
            if is_gpu_idle(gpu) and gpu['memory_free'] >= args.min_memory:
                memory_to_occupy = int(gpu['memory_free'] * args.ratio)
                if memory_to_occupy >= args.min_memory:
                    occupy_gpu_memory(gpu['index'], memory_to_occupy, args.duration)
    else:
        # Continuous monitoring
        monitor_and_occupy(
            occupancy_ratio=args.ratio,
            check_interval=args.interval,
            max_duration=args.duration,
            min_free_memory=args.min_memory
        )

if __name__ == '__main__':
    main()
