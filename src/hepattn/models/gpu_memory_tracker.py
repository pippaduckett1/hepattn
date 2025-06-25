#!/usr/bin/env python3
"""
Simple GPU Memory Tracker for PyTorch experiments.
Tracks peak GPU memory usage during training/inference.
"""

import torch
import time
from typing import Dict, Optional, Any
from contextlib import contextmanager
import json
from pathlib import Path


class GPUMemoryTracker:
    """
    Simple GPU memory tracker for PyTorch experiments.
    
    Usage:
        tracker = GPUMemoryTracker()
        
        # Track a single operation
        with tracker.track("forward_pass"):
            output = model(input_data)
        
        # Track multiple operations
        tracker.start_tracking("training_epoch")
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
        tracker.stop_tracking("training_epoch")
        
        # Get results
        print(tracker.get_summary())
        tracker.save_results("memory_log.json")
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the GPU memory tracker.
        
        Args:
            device: GPU device to track (e.g., 'cuda:0'). If None, uses current device.
        """
        self.device = device or (torch.cuda.current_device() if torch.cuda.is_available() else None)
        self.tracking_data: Dict[str, Dict[str, Any]] = {}
        self.active_tracks: Dict[str, Dict[str, Any]] = {}
        
        # Reset peak memory stats at initialization
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics in MB."""
        if not torch.cuda.is_available():
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'max_allocated': 0.0,
                'max_reserved': 0.0
            }
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'max_reserved': torch.cuda.max_memory_reserved() / 1024 / 1024
        }
    
    def start_tracking(self, name: str) -> None:
        """
        Start tracking memory usage for a named operation.
        
        Args:
            name: Name of the operation to track
        """
        if name in self.active_tracks:
            print(f"Warning: Already tracking '{name}'. Stopping previous track.")
            self.stop_tracking(name)
        
        initial_stats = self._get_memory_stats()
        self.active_tracks[name] = {
            'start_time': time.time(),
            'initial_stats': initial_stats
        }
        
        print(f"Started tracking '{name}' - Initial GPU memory: {initial_stats['allocated']:.2f} MB")
    
    def stop_tracking(self, name: str) -> None:
        """
        Stop tracking memory usage for a named operation.
        
        Args:
            name: Name of the operation to stop tracking
        """
        if name not in self.active_tracks:
            print(f"Warning: No active tracking for '{name}'")
            return
        
        end_stats = self._get_memory_stats()
        track_data = self.active_tracks[name]
        
        duration = time.time() - track_data['start_time']
        initial_stats = track_data['initial_stats']
        
        # Calculate memory changes
        memory_data = {
            'duration': duration,
            'initial_allocated': initial_stats['allocated'],
            'initial_reserved': initial_stats['reserved'],
            'final_allocated': end_stats['allocated'],
            'final_reserved': end_stats['reserved'],
            'peak_allocated': end_stats['max_allocated'],
            'peak_reserved': end_stats['max_reserved'],
            'allocated_increase': end_stats['allocated'] - initial_stats['allocated'],
            'reserved_increase': end_stats['reserved'] - initial_stats['reserved'],
            'peak_allocated_increase': end_stats['max_allocated'] - initial_stats['allocated'],
            'peak_reserved_increase': end_stats['max_reserved'] - initial_stats['reserved']
        }
        
        self.tracking_data[name] = memory_data
        
        print(f"Stopped tracking '{name}' - Duration: {duration:.3f}s, "
              f"Peak GPU memory: {end_stats['max_allocated']:.2f} MB "
              f"(+{memory_data['peak_allocated_increase']:.2f} MB)")
        
        del self.active_tracks[name]
    
    @contextmanager
    def track(self, name: str):
        """
        Context manager for tracking memory usage.
        
        Args:
            name: Name of the operation to track
            
        Usage:
            with tracker.track("forward_pass"):
                output = model(input_data)
        """
        self.start_tracking(name)
        try:
            yield
        finally:
            self.stop_tracking(name)
    
    def get_current_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        return self._get_memory_stats()
    
    def get_track_data(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tracking data for a specific operation."""
        return self.tracking_data.get(name)
    
    def get_summary(self) -> str:
        """Get a formatted summary of all tracked operations."""
        if not self.tracking_data:
            return "No tracking data available."
        
        summary = ["GPU Memory Tracking Summary", "=" * 40]
        
        for name, data in self.tracking_data.items():
            summary.append(f"\n{name}:")
            summary.append(f"  Duration: {data['duration']:.3f}s")
            summary.append(f"  Peak GPU Memory: {data['peak_allocated']:.2f} MB")
            summary.append(f"  Memory Increase: +{data['peak_allocated_increase']:.2f} MB")
            summary.append(f"  Peak Reserved: {data['peak_reserved']:.2f} MB")
        
        # Add overall peak
        if torch.cuda.is_available():
            current_stats = self._get_memory_stats()
            summary.append(f"\nOverall Peak GPU Memory: {current_stats['max_allocated']:.2f} MB")
            summary.append(f"Overall Peak Reserved: {current_stats['max_reserved']:.2f} MB")
        
        return "\n".join(summary)
    
    def save_results(self, filepath: str) -> None:
        """
        Save tracking results to a JSON file.
        
        Args:
            filepath: Path to save the results
        """
        # Add overall stats
        results = {
            'tracking_data': self.tracking_data,
            'overall_stats': self._get_memory_stats(),
            'timestamp': time.time(),
            'formatted_summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def reset(self) -> None:
        """Reset all tracking data and peak memory stats."""
        self.tracking_data.clear()
        self.active_tracks.clear()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        print("Memory tracker reset")


# Convenience functions for quick memory tracking
def track_gpu_memory(name: str = "operation"):
    """
    Simple decorator to track GPU memory usage of a function.
    
    Usage:
        @track_gpu_memory("model_forward")
        def forward_pass(model, data):
            return model(data)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = GPUMemoryTracker()
            with tracker.track(name):
                result = func(*args, **kwargs)
            print(tracker.get_summary())
            return result
        return wrapper
    return decorator


def quick_memory_check() -> Dict[str, float]:
    """
    Quick function to get current GPU memory usage.
    
    Returns:
        Dictionary with current memory stats
    """
    tracker = GPUMemoryTracker()
    return tracker.get_current_memory()


# Example usage
if __name__ == "__main__":
    # Example 1: Using context manager
    tracker = GPUMemoryTracker()
    
    print("Example 1: Context manager usage")
    with tracker.track("tensor_creation"):
        # Create some tensors
        x = torch.randn(1000, 1000, device='cuda' if torch.cuda.is_available() else 'cpu')
        y = torch.randn(1000, 1000, device='cuda' if torch.cuda.is_available() else 'cpu')
        z = torch.mm(x, y)
    
    # Example 2: Manual start/stop
    print("\nExample 2: Manual start/stop")
    tracker.start_tracking("matrix_ops")
    for i in range(5):
        a = torch.randn(500, 500, device='cuda' if torch.cuda.is_available() else 'cpu')
        b = torch.randn(500, 500, device='cuda' if torch.cuda.is_available() else 'cpu')
        c = torch.mm(a, b)
        del a, b, c  # Clean up
    tracker.stop_tracking("matrix_ops")
    
    # Print summary
    print("\n" + tracker.get_summary())
    
    # Save results
    tracker.save_results("example_memory_log.json")