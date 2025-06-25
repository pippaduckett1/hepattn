#!/usr/bin/env python3
"""
Memory profiling script to compare full mask loss vs point-sampled mask loss.
Uses actual TrackML dataset for realistic measurements.
"""

import torch
import torch.nn.functional as F
import time
import psutil
import os
from memory_profiler import profile
import gc
import sys
from pathlib import Path

# Add the src directory to the path so we can import hepattn modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the loss functions
from hepattn.models.loss import (
    mask_dice_loss, 
    mask_focal_loss,
    point_sampled_mask_dice_loss,
    point_sampled_mask_focal_loss
)

# Import the dataset
from hepattn.experiments.trackml.data import TrackMLDataset

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024
    
    # Get GPU memory if available
    gpu_memory = 0
    gpu_peak_allocated = 0
    gpu_peak_reserved = 0
    if torch.cuda.is_available():
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_peak_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        gpu_peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    
    return cpu_memory, gpu_memory, gpu_peak_allocated, gpu_peak_reserved


def get_detailed_memory_usage():
    """Get detailed memory usage information."""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024
    
    gpu_info = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_info = {
            'allocated': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'max_reserved': torch.cuda.max_memory_reserved() / 1024 / 1024,
        }
    
    return cpu_memory, gpu_info

def load_real_dataset(num_events=5, data_dir="/share/rcifdata/pduckett/hepattn/data/trackml/prepped/val/"):
    """Load real TrackML dataset for testing."""
    print(f"Loading real TrackML dataset from {data_dir}")
    print(f"Number of events: {num_events}")
    
    # Define inputs and targets based on your config
    inputs = {
        "hit": [
            "x", "y", "z", "r", "s", "eta", "phi", "u", "v",
            "charge_frac", "leta", "lphi", "lx", "ly", "lz", "geta", "gphi"
        ]
    }
    
    targets = {
        "particle": ["pt", "eta", "phi"]
    }
    
    # Create dataset
    dataset = TrackMLDataset(
        dirpath=data_dir,
        inputs=inputs,
        targets=targets,
        num_events=num_events,
        hit_volume_ids=[8],  # pixel barrel
        particle_min_pt=1.0,
        particle_max_abs_eta=4.0,
        particle_min_num_hits=3,
        event_max_num_particles=2000,
    )
    
    print(f"Dataset created with {len(dataset)} events")
    
    # Load a few events to get realistic data
    events = []
    for i in range(min(num_events, len(dataset))):
        inputs, targets = dataset[i]
        events.append((inputs, targets))
        print(f"Loaded event {i}: {inputs['hit_x'].shape[1]} hits, {targets['particle_valid'].sum().item()} particles")
    
    return events


def verify_device_placement(pred_logits, targets, mask, weight, device):
    """Verify that all tensors are on the correct device."""
    print(f"Device verification (expected: {device}):")
    print(f"  pred_logits: {pred_logits.device}")
    print(f"  targets: {targets.device}")
    print(f"  mask: {mask.device}")
    print(f"  weight: {weight.device}")
    
    # Check if all tensors are on the same device
    tensors = [pred_logits, targets, mask, weight]
    devices = [t.device for t in tensors]
    print(devices)
    
    if len(set(devices)) > 1:
        raise RuntimeError(f"Tensors are on different devices: {devices}")
    elif devices[0] != torch.device(device):
        raise RuntimeError(f"Tensors are on {devices[0]} but expected {device}")
    else:
        print("✓ All tensors are on the correct device")


def create_test_data(batch_size=2, num_objects=10, num_points_per_track_per_track=1000, device='cuda'):
    """Create test data for mask loss computation using real dataset."""
    print(f"Creating test data from real dataset: batch_size={batch_size}, num_objects={num_objects}, num_points_per_track_per_track={num_points_per_track_per_track}")
    
    # Load real dataset
    events = load_real_dataset(num_events=batch_size)
    
    # Use the first event as our test case
    inputs, targets = events[0]
    
    # Extract the relevant tensors for mask loss testing
    num_hits = inputs['hit_x'].shape[1]
    num_particles = targets['particle_valid'].shape[1]
    
    print(f"Real data: {num_hits} hits, {num_particles} particles")
    
    # Create simulated model outputs (logits for track-hit assignments)
    # Shape: [batch_size, num_particles, num_hits]

    pred_logits = torch.randn(batch_size, num_particles, num_hits, device=device)
    
    # Get the real targets and move to correct device
    targets_tensor = targets['particle_hit_valid'].float().to(device)  # [1, num_particles, num_hits]
    
    # Get the mask (which hits and particles are valid) and move to correct device
    hit_valid = inputs['hit_valid'].to(device)  # [1, num_hits]
    particle_valid = targets['particle_valid'].to(device)  # [1, num_particles]
    
    # Create the object-hit mask (both object and hit must be valid)
    object_hit_mask = particle_valid.unsqueeze(-1) & hit_valid.unsqueeze(-2)  # [1, num_particles, num_hits]
    
    # Create weight tensor
    weight = targets_tensor + 0.1 * (1 - targets_tensor)  # Higher weight for positive samples
    
    # Expand to batch size if needed
    if batch_size > 1:
        pred_logits = pred_logits.expand(batch_size, -1, -1)
        targets_tensor = targets_tensor.expand(batch_size, -1, -1)
        object_hit_mask = object_hit_mask.expand(batch_size, -1, -1)
        weight = weight.expand(batch_size, -1, -1)
    
    # Verify device placement
    verify_device_placement(pred_logits, targets_tensor, object_hit_mask, weight, device)
    
    return pred_logits, targets_tensor, object_hit_mask, weight


def time_and_memory_test(func, name, *args, **kwargs):
    """Test a function and measure time and memory usage."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    
    # Clear cache and garbage collect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Measure initial memory
    initial_cpu, initial_gpu_info = get_detailed_memory_usage()
    print(f"Initial memory - CPU: {initial_cpu:.2f} MB")
    if initial_gpu_info:
        print(f"Initial GPU - Allocated: {initial_gpu_info['allocated']:.2f} MB, Reserved: {initial_gpu_info['reserved']:.2f} MB")
    
    # Time the function
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Force computation to ensure memory is actually allocated
    if hasattr(result, 'numel'):
        if result.numel() == 1:
            # Scalar tensor
            result_value = result.item()
        else:
            # Multi-element tensor - compute mean to get a scalar
            result_value = result.mean().item()
            print(f"Warning: Result has {result.numel()} elements, using mean: {result_value:.6f}")
    elif hasattr(result, 'cpu'):
        # Non-tensor object with cpu method
        result.cpu()
        result_value = float(result) if hasattr(result, '__float__') else 0.0
    else:
        # Other type
        result_value = float(result) if hasattr(result, '__float__') else 0.0
    
    # Measure peak memory after forcing computation
    peak_cpu, peak_gpu_info = get_detailed_memory_usage()
    print(f"Peak memory - CPU: {peak_cpu:.2f} MB")
    if peak_gpu_info:
        print(f"Peak GPU - Allocated: {peak_gpu_info['allocated']:.2f} MB, Reserved: {peak_gpu_info['reserved']:.2f} MB")
        print(f"Max GPU - Allocated: {peak_gpu_info['max_allocated']:.2f} MB, Reserved: {peak_gpu_info['max_reserved']:.2f} MB")
    
    print(f"Memory increase - CPU: {peak_cpu - initial_cpu:.2f} MB")
    if initial_gpu_info and peak_gpu_info:
        print(f"Memory increase - GPU: {peak_gpu_info['allocated'] - initial_gpu_info['allocated']:.2f} MB")
        print(f"Max memory increase - GPU: {peak_gpu_info['max_allocated'] - initial_gpu_info['allocated']:.2f} MB")
    
    print(f"Time: {end_time - start_time:.4f} seconds")
    print(f"Result: {result_value:.6f}")
    
    # Additional debugging: check if result is a scalar
    if hasattr(result, 'shape'):
        print(f"Result type: {type(result)}, shape: {result.shape}, numel: {result.numel()}")
    else:
        print(f"Result type: {type(result)}")
    
    # Additional debugging: check if result is a scalar
    print(f"Result type: {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")

    # Clear cache and garbage collect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Return memory increase (use max allocated for GPU to capture peak usage)
    gpu_memory_increase = 0
    if initial_gpu_info and peak_gpu_info:
        gpu_memory_increase = peak_gpu_info['max_allocated'] - initial_gpu_info['allocated']
    
    return result, end_time - start_time, peak_cpu - initial_cpu, gpu_memory_increase


def debug_tensor_sizes(pred_logits, targets, mask, weight):
    """Debug function to print tensor sizes and memory usage."""
    print(f"\nTensor Size Debug:")
    print(f"  pred_logits: {pred_logits.shape} ({pred_logits.numel()} elements)")
    print(f"  targets: {targets.shape} ({targets.numel()} elements)")
    print(f"  mask: {mask.shape} ({mask.numel()} elements)")
    print(f"  weight: {weight.shape} ({weight.numel()} elements)")
    
    # Calculate memory usage in MB
    pred_memory = pred_logits.numel() * pred_logits.element_size() / 1024 / 1024
    targets_memory = targets.numel() * targets.element_size() / 1024 / 1024
    mask_memory = mask.numel() * mask.element_size() / 1024 / 1024
    weight_memory = weight.numel() * weight.element_size() / 1024 / 1024
    
    print(f"  Memory usage:")
    print(f"    pred_logits: {pred_memory:.2f} MB")
    print(f"    targets: {targets_memory:.2f} MB")
    print(f"    mask: {mask_memory:.2f} MB")
    print(f"    weight: {weight_memory:.2f} MB")
    print(f"    Total: {pred_memory + targets_memory + mask_memory + weight_memory:.2f} MB")
    
    # Check if tensors are on GPU
    print(f"  Device info:")
    print(f"    pred_logits: {pred_logits.device}")
    print(f"    targets: {targets.device}")
    print(f"    mask: {mask.device}")
    print(f"    weight: {weight.device}")


def test_without_mask():
    """Test loss functions without mask to see if that fixes the issue."""
    print(f"\n{'='*50}")
    print("TESTING WITHOUT MASK")
    print(f"{'='*50}")
    
    # Create simple test data without mask
    batch_size, num_objects, num_points_per_track = 1, 10, 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pred_logits = torch.randn(batch_size, num_objects, num_points_per_track, device=device)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_points_per_track), dtype=torch.float, device=device)
    weight = torch.ones(batch_size, num_objects, num_points_per_track, device=device)
    
    print(f"Test data: {batch_size} batch, {num_objects} objects, {num_points_per_track} points")
    print(f"Input shapes:")
    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  weight: {weight.shape}")
    
    # Test full dice loss without mask
    print(f"\nTesting full dice loss (no mask):")
    try:
        full_dice = mask_dice_loss(pred_logits, targets, weight=weight)
        print(f"  Full dice result shape: {full_dice.shape}, numel: {full_dice.numel()}")
        print(f"  Full dice value: {full_dice.item() if full_dice.numel() == 1 else 'MULTI-ELEMENT'}")
    except Exception as e:
        print(f"  Full dice error: {e}")
    
    # Test sampled dice loss without mask
    print(f"\nTesting sampled dice loss (no mask):")
    try:
        sampled_dice = point_sampled_mask_dice_loss(pred_logits, targets, weight=weight, num_points_per_track=20)
        print(f"  Sampled dice result shape: {sampled_dice.shape}, numel: {sampled_dice.numel()}")
        print(f"  Sampled dice value: {sampled_dice.item() if sampled_dice.numel() == 1 else 'MULTI-ELEMENT'}")
    except Exception as e:
        print(f"  Sampled dice error: {e}")
    
    # Test full focal loss without mask
    print(f"\nTesting full focal loss (no mask):")
    try:
        full_focal = mask_focal_loss(pred_logits, targets, weight=weight)
        print(f"  Full focal result shape: {full_focal.shape}, numel: {full_focal.numel()}")
        print(f"  Full focal value: {full_focal.item() if full_focal.numel() == 1 else 'MULTI-ELEMENT'}")
    except Exception as e:
        print(f"  Full focal error: {e}")
    
    # Test sampled focal loss without mask
    print(f"\nTesting sampled focal loss (no mask):")
    try:
        sampled_focal = point_sampled_mask_focal_loss(pred_logits, targets, weight=weight, num_points_per_track=20)
        print(f"  Sampled focal result shape: {sampled_focal.shape}, numel: {sampled_focal.numel()}")
        print(f"  Sampled focal value: {sampled_focal.item() if sampled_focal.numel() == 1 else 'MULTI-ELEMENT'}")
    except Exception as e:
        print(f"  Sampled focal error: {e}")

def test_memory_profiling_works(device):
    """Simple test to verify memory profiling is working."""
    print(f"\n{'='*50}")
    print("VERIFYING MEMORY PROFILING WORKS")
    print(f"{'='*50}")
    
    def create_large_tensor(size_mb=100):
        """Create a tensor of specified size in MB."""
        size_bytes = size_mb * 1024 * 1024
        num_elements = size_bytes // 4  # float32 = 4 bytes
        return torch.randn(num_elements, device=device)
    
    # Test with a simple tensor creation
    result, time_taken, cpu_increase, gpu_increase = time_and_memory_test(
        create_large_tensor, 
        "Create 100MB Tensor",
        100
    )
    
    print(f"\nMemory profiling test results:")
    print(f"CPU memory increase: {cpu_increase:.2f} MB")
    print(f"GPU memory increase: {gpu_increase:.2f} MB")
    
    if gpu_increase > 50:  # Should see significant GPU memory increase
        print("✓ GPU memory profiling is working correctly")
    else:
        print("⚠ GPU memory profiling may not be working correctly")
    
    if cpu_increase > 10:  # Should see some CPU memory increase
        print("✓ CPU memory profiling is working correctly")
    else:
        print("⚠ CPU memory profiling may not be working correctly")


def test_large_tensor_memory(device):
    """Test memory usage with larger tensors to see if differences are observable."""
    print(f"\n{'='*50}")
    print("LARGE TENSOR MEMORY TEST")
    print(f"{'='*50}")
    
    # Create larger test data
    batch_size, num_objects, num_points_per_track = 1, 100, 10000  # Much larger tensors
    
    print(f"Creating large test data: {batch_size} batch, {num_objects} objects, {num_points_per_track} points")
    print(f"Total elements: {batch_size * num_objects * num_points_per_track:,}")
    
    pred_logits = torch.randn(batch_size, num_objects, num_points_per_track, device=device)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_points_per_track), dtype=torch.float, device=device)
    mask = torch.ones(batch_size, num_objects, num_points_per_track, dtype=torch.bool, device=device)
    weight = torch.ones(batch_size, num_objects, num_points_per_track, device=device)
    
    # Debug tensor sizes
    debug_tensor_sizes(pred_logits, targets, mask, weight)
    
    # Test full dice loss
    result, time_taken, cpu_increase, gpu_increase = time_and_memory_test(
        mask_dice_loss, 
        "Full Dice Loss (Large)",
        pred_logits, targets, mask=mask, weight=weight
    )
    print(f"Full dice loss result: {result.item():.6f}")
    print(f"Full dice loss - CPU increase: {cpu_increase:.2f} MB, GPU increase: {gpu_increase:.2f} MB")
    
    # Test sampled dice loss with different numbers of points
    for num_sampled_points in [10, 50, 100]:
        result, time_taken, cpu_increase, gpu_increase = time_and_memory_test(
            point_sampled_mask_dice_loss,
            f"Sampled Dice Loss ({num_sampled_points} points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=num_sampled_points
        )
        print(f"Sampled dice loss ({num_sampled_points} points) result: {result.item():.6f}")
        print(f"Sampled dice loss ({num_sampled_points} points) - CPU increase: {cpu_increase:.2f} MB, GPU increase: {gpu_increase:.2f} MB")


def test_loss_function_comparison(device):
    """Simple test to compare full vs sampled loss functions."""
    print(f"\n{'='*50}")
    print("LOSS FUNCTION COMPARISON TEST")
    print(f"{'='*50}")
    
    # Create simple test data
    batch_size, num_objects, num_points_per_track = 1, 10, 100
    
    pred_logits = torch.randn(batch_size, num_objects, num_points_per_track, device=device)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_points_per_track), dtype=torch.float, device=device)
    mask = torch.ones(batch_size, num_objects, num_points_per_track, dtype=torch.bool, device=device)
    weight = torch.ones(batch_size, num_objects, num_points_per_track, device=device)
    
    print(f"Test data: {batch_size} batch, {num_objects} objects, {num_points_per_track} points")
    
    # Test full dice loss
    full_dice = mask_dice_loss(pred_logits, targets, mask=mask, weight=weight)
    print(f"Full dice loss: {full_dice.item():.6f}")
    
    # Test sampled dice loss with different numbers of points
    sampled_dice_5 = point_sampled_mask_dice_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=5)
    sampled_dice_10 = point_sampled_mask_dice_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=10)
    sampled_dice_20 = point_sampled_mask_dice_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=20)
    
    print(f"Sampled dice loss (5 points): {sampled_dice_5.item():.6f}")
    print(f"Sampled dice loss (10 points): {sampled_dice_10.item():.6f}")
    print(f"Sampled dice loss (20 points): {sampled_dice_20.item():.6f}")
    
    # Test full focal loss
    full_focal = mask_focal_loss(pred_logits, targets, mask=mask, weight=weight)
    print(f"Full focal loss: {full_focal.item():.6f}")
    
    # Test sampled focal loss
    sampled_focal_5 = point_sampled_mask_focal_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=5)
    sampled_focal_10 = point_sampled_mask_focal_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=10)
    sampled_focal_20 = point_sampled_mask_focal_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=20)
    
    print(f"Sampled focal loss (5 points): {sampled_focal_5.item():.6f}")
    print(f"Sampled focal loss (10 points): {sampled_focal_10.item():.6f}")
    print(f"Sampled focal loss (20 points): {sampled_focal_20.item():.6f}")
    
    # Check if results are different
    print(f"\nResults comparison:")
    print(f"Dice loss difference (5 points): {abs(full_dice.item() - sampled_dice_5.item()):.6f}")
    print(f"Dice loss difference (10 points): {abs(full_dice.item() - sampled_dice_10.item()):.6f}")
    print(f"Dice loss difference (20 points): {abs(full_dice.item() - sampled_dice_20.item()):.6f}")
    
    print(f"Focal loss difference (5 points): {abs(full_focal.item() - sampled_focal_5.item()):.6f}")
    print(f"Focal loss difference (10 points): {abs(full_focal.item() - sampled_focal_10.item()):.6f}")
    print(f"Focal loss difference (20 points): {abs(full_focal.item() - sampled_focal_20.item()):.6f}")


def test_full_vs_sampled_losses(device):
    """Compare full mask losses vs point-sampled versions using real data."""
    
    # Test parameters - using realistic sizes based on your dataset
    test_configs = [
        {"batch_size": 1, "num_objects": 100, "num_points_per_track_per_track": 500, "name": "Small Event"},
        {"batch_size": 1, "num_objects": 500, "num_points_per_track_per_track": 1000, "name": "Medium Event"},
        {"batch_size": 1, "num_objects": 1000, "num_points_per_track_per_track": 2000, "name": "Large Event"},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n{'#'*60}")
        print(f"Testing {config['name']} configuration")
        print(f"{'#'*60}")
        
        # Create test data from real dataset
        pred_logits, targets, mask, weight = create_test_data(
            batch_size=config['batch_size'],
            num_objects=config['num_objects'], 
            num_points_per_track_per_track=config['num_points_per_track_per_track'],
            device=device
        )
        
        # Debug tensor sizes
        debug_tensor_sizes(pred_logits, targets, mask, weight)
        
        config_results = {}
        
        # Test full dice loss
        result, time_taken, memory_used_cpu, memory_used_gpu = time_and_memory_test(
            mask_dice_loss, 
            "Full Dice Loss",
            pred_logits, targets, mask=mask, weight=weight
        )
        config_results['full_dice'] = {'time': time_taken, 'memory_cpu': memory_used_cpu, 'memory_gpu': memory_used_gpu, 'result': result.item()}
        
        # Test point-sampled dice loss
        result, time_taken, memory_used_cpu, memory_used_gpu = time_and_memory_test(
            point_sampled_mask_dice_loss,
            "Point-sampled Dice Loss (20 points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=20
        )
        config_results['sampled_dice_20'] = {'time': time_taken, 'memory_cpu': memory_used_cpu, 'memory_gpu': memory_used_gpu, 'result': result.item()}
        
        # Test point-sampled dice loss with more points
        result, time_taken, memory_used_cpu, memory_used_gpu = time_and_memory_test(
            point_sampled_mask_dice_loss,
            "Point-sampled Dice Loss (50 points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=50
        )
        config_results['sampled_dice_50'] = {'time': time_taken, 'memory_cpu': memory_used_cpu, 'memory_gpu': memory_used_gpu, 'result': result.item()}
        
        # Test full focal loss
        result, time_taken, memory_used_cpu, memory_used_gpu = time_and_memory_test(
            mask_focal_loss,
            "Full Focal Loss",
            pred_logits, targets, mask=mask, weight=weight
        )
        config_results['full_focal'] = {'time': time_taken, 'memory_cpu': memory_used_cpu, 'memory_gpu': memory_used_gpu, 'result': result.item()}
        
        # Test point-sampled focal loss
        result, time_taken, memory_used_cpu, memory_used_gpu = time_and_memory_test(
            point_sampled_mask_focal_loss,
            "Point-sampled Focal Loss (20 points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=20
        )
        config_results['sampled_focal_20'] = {'time': time_taken, 'memory_cpu': memory_used_cpu, 'memory_gpu': memory_used_gpu, 'result': result.item()}
        
        # Test point-sampled focal loss with more points
        result, time_taken, memory_used_cpu, memory_used_gpu = time_and_memory_test(
            point_sampled_mask_focal_loss,
            "Point-sampled Focal Loss (50 points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=50
        )
        config_results['sampled_focal_50'] = {'time': time_taken, 'memory_cpu': memory_used_cpu, 'memory_gpu': memory_used_gpu, 'result': result.item()}
        
        results[config['name']] = config_results
    
    return results



def debug_memory_issue(device):
    """Debug function to understand why memory changes might not be visible."""
    print(f"\n{'='*50}")
    print("DEBUGGING MEMORY ISSUE")
    print(f"{'='*50}")
    
    # Check if CUDA is available and set device explicitly
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = 'cpu'
    else:
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear GPU memory and reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Create a simple test case
    batch_size, num_objects, num_points = 1, 10, 1000
    
    print(f"Creating test tensors: {batch_size} x {num_objects} x {num_points}")
    print(f"Total elements: {batch_size * num_objects * num_points:,}")
    
    pred_logits = torch.randn(batch_size, num_objects, num_points, device=device)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_points), dtype=torch.float, device=device)
    weight = torch.ones(batch_size, num_objects, num_points, device=device)

    # Verify all tensors are on the correct device
    print(f"Tensor devices:")
    print(f"  pred_logits: {pred_logits.device}")
    print(f"  targets: {targets.device}")
    print(f"  weight: {weight.device}")
    
    # Calculate memory usage
    total_memory_mb = (pred_logits.numel() + targets.numel() + weight.numel()) * 4 / 1024 / 1024
    print(f"Input tensor memory: {total_memory_mb:.2f} MB")
    
    # Test full dice loss
    print(f"\nTesting full dice loss:")
    initial_cpu, initial_gpu, _, _ = get_memory_usage()
    full_dice = mask_dice_loss(pred_logits, targets, weight=weight)
    peak_cpu, peak_gpu, peak_gpu_allocated, peak_gpu_reserved = get_memory_usage()
    print(f"  Full dice result: {full_dice.item():.6f}")
    print(f"  Memory change - CPU: {peak_cpu - initial_cpu:.2f} MB, GPU: {peak_gpu - initial_gpu:.2f} MB")
    
    # Test sampled dice loss
    print(f"\nTesting sampled dice loss:")
    initial_cpu, initial_gpu, _, _ = get_memory_usage()
    sampled_dice = point_sampled_mask_dice_loss(pred_logits, targets, weight=weight, num_points_per_track=20)
    peak_cpu, peak_gpu, peak_gpu_allocated, peak_gpu_reserved = get_memory_usage()
    print(f"  Sampled dice result: {sampled_dice.item():.6f}")
    print(f"  Memory change - CPU: {peak_cpu - initial_cpu:.2f} MB, GPU: {peak_gpu - initial_gpu:.2f} MB")
    
    # Test with much larger tensors
    print(f"\nTesting with larger tensors:")
    batch_size, num_objects, num_points = 1, 100, 10000
    print(f"Creating large test tensors: {batch_size} x {num_objects} x {num_points}")
    print(f"Total elements: {batch_size * num_objects * num_points:,}")
    
    pred_logits_large = torch.randn(batch_size, num_objects, num_points, device=device)
    targets_large = torch.randint(0, 2, (batch_size, num_objects, num_points), dtype=torch.float, device=device)
    weight_large = torch.ones(batch_size, num_objects, num_points, device=device)
    
    # Verify large tensors are on the correct device
    print(f"Large tensor devices:")
    print(f"  pred_logits_large: {pred_logits_large.device}")
    print(f"  targets_large: {targets_large.device}")
    print(f"  weight_large: {weight_large.device}")

    total_memory_mb_large = (pred_logits_large.numel() + targets_large.numel() + weight_large.numel()) * 4 / 1024 / 1024
    print(f"Large input tensor memory: {total_memory_mb_large:.2f} MB")
    
    # Test full dice loss with large tensors
    print(f"\nTesting full dice loss (large):")
    initial_cpu, initial_gpu, _, _ = get_memory_usage()
    full_dice_large = mask_dice_loss(pred_logits_large, targets_large, weight=weight_large)
    peak_cpu, peak_gpu, peak_gpu_allocated, peak_gpu_reserved = get_memory_usage()
    print(f"  Full dice result: {full_dice_large.item():.6f}")
    print(f"  Memory change - CPU: {peak_cpu - initial_cpu:.2f} MB, GPU: {peak_gpu - initial_gpu:.2f} MB")
    print(f"  Peak GPU - Allocated: {peak_gpu_allocated:.2f} MB, Reserved: {peak_gpu_reserved:.2f} MB")
    
    # Test sampled dice loss with large tensors
    print(f"\nTesting sampled dice loss (large):")
    initial_cpu, initial_gpu, _, _ = get_memory_usage()
    sampled_dice_large = point_sampled_mask_dice_loss(pred_logits_large, targets_large, weight=weight_large, num_points_per_track=20)
    peak_cpu, peak_gpu, peak_gpu_allocated, peak_gpu_reserved = get_memory_usage()
    print(f"  Sampled dice result: {sampled_dice_large.item():.6f}")
    print(f"  Memory change - CPU: {peak_cpu - initial_cpu:.2f} MB, GPU: {peak_gpu - initial_gpu:.2f} MB")
    print(f"  Peak GPU - Allocated: {peak_gpu_allocated:.2f} MB, Reserved: {peak_gpu_reserved:.2f} MB")


    # If using GPU, show peak memory usage
    if device != 'cpu':
        print(f"\nPeak GPU memory usage:")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"  Max reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    
    # Clean up
    del pred_logits, targets, weight, full_dice, sampled_dice
    del pred_logits_large, targets_large, weight_large, full_dice_large, sampled_dice_large
    if device == 'cuda':
        torch.cuda.empty_cache()


def print_summary(results):
    """Print a summary of the results."""
    print(f"\n{'='*80}")
    print("MEMORY AND TIME COMPARISON SUMMARY (REAL DATA)")
    print(f"{'='*80}")
    
    for config_name, config_results in results.items():
        print(f"\n{config_name} Configuration:")
        print("-" * 40)
        
        # Dice loss comparison
        full_dice = config_results['full_dice']
        sampled_dice_20 = config_results['sampled_dice_20']
        sampled_dice_50 = config_results['sampled_dice_50']
        
        print(f"Dice Loss:")
        print(f"  Full:           {full_dice['memory_cpu']:6.2f} MB, {full_dice['time']:6.4f}s")
        print(f"  Sampled (20):   {sampled_dice_20['memory_cpu']:6.2f} MB, {sampled_dice_20['time']:6.4f}s")
        print(f"  Sampled (50):   {sampled_dice_50['memory_cpu']:6.2f} MB, {sampled_dice_50['time']:6.4f}s")
        # print(f"  Memory savings (20): {(1 - sampled_dice_20['memory_cpu']/full_dice['memory_cpu'])*100:5.1f}%")
        # print(f"  Memory savings (50): {(1 - sampled_dice_50['memory_cpu']/full_dice['memory_cpu'])*100:5.1f}%")
        print(f"  Speedup (20):   {full_dice['time']/sampled_dice_20['time']:5.1f}x")
        print(f"  Speedup (50):   {full_dice['time']/sampled_dice_50['time']:5.1f}x")
        
        # Focal loss comparison
        full_focal = config_results['full_focal']
        sampled_focal_20 = config_results['sampled_focal_20']
        sampled_focal_50 = config_results['sampled_focal_50']
        
        print(f"Focal Loss:")
        print(f"  Full:           {full_focal['memory_cpu']:6.2f} MB, {full_focal['time']:6.4f}s")
        print(f"  Sampled (20):   {sampled_focal_20['memory_cpu']:6.2f} MB, {sampled_focal_20['time']:6.4f}s")
        print(f"  Sampled (50):   {sampled_focal_50['memory_cpu']:6.2f} MB, {sampled_focal_50['time']:6.4f}s")
        # print(f"  Memory savings (20): {(1 - sampled_focal_20['memory_cpu']/full_focal['memory_cpu'])*100:5.1f}%")
        # print(f"  Memory savings (50): {(1 - sampled_focal_50['memory_cpu']/full_focal['memory_cpu'])*100:5.1f}%")
        print(f"  Speedup (20):   {full_focal['time']/sampled_focal_20['time']:5.1f}x")
        print(f"  Speedup (50):   {full_focal['time']/sampled_focal_50['time']:5.1f}x")


def test_accuracy_comparison(device):
    """Test if the point-sampled losses give similar results to full losses."""
    print(f"\n{'='*60}")
    print("ACCURACY COMPARISON (REAL DATA)")
    print(f"{'='*60}")
    
    # Create test data from real dataset
    pred_logits, targets, mask, weight = create_test_data(
        batch_size=1, num_objects=500, num_points_per_track_per_track=1000, device=device
    )
    
    print(f"Test data shapes:")
    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  weight: {weight.shape}")
    
    # Compute full losses
    full_dice = mask_dice_loss(pred_logits, targets, mask=mask, weight=weight)
    full_focal = mask_focal_loss(pred_logits, targets, mask=mask, weight=weight)
    
    # Compute sampled losses with different numbers of points
    sampled_dice_10 = point_sampled_mask_dice_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=10)
    sampled_dice_20 = point_sampled_mask_dice_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=20)
    sampled_dice_50 = point_sampled_mask_dice_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=50)
    
    sampled_focal_10 = point_sampled_mask_focal_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=10)
    sampled_focal_20 = point_sampled_mask_focal_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=20)
    sampled_focal_50 = point_sampled_mask_focal_loss(pred_logits, targets, mask=mask, weight=weight, num_points_per_track=50)
    
    print(f"Dice Loss Comparison:")
    print(f"  Full:     {full_dice.item():.6f}")
    print(f"  Sampled 10: {sampled_dice_10.item():.6f} (diff: {abs(full_dice.item() - sampled_dice_10.item()):.6f})")
    print(f"  Sampled 20: {sampled_dice_20.item():.6f} (diff: {abs(full_dice.item() - sampled_dice_20.item()):.6f})")
    print(f"  Sampled 50: {sampled_dice_50.item():.6f} (diff: {abs(full_dice.item() - sampled_dice_50.item()):.6f})")
    
    print(f"\nFocal Loss Comparison:")
    print(f"  Full:     {full_focal.item():.6f}")
    print(f"  Sampled 10: {sampled_focal_10.item():.6f} (diff: {abs(full_focal.item() - sampled_focal_10.item()):.6f})")
    print(f"  Sampled 20: {sampled_focal_20.item():.6f} (diff: {abs(full_focal.item() - sampled_focal_20.item()):.6f})")
    print(f"  Sampled 50: {sampled_focal_50.item():.6f} (diff: {abs(full_focal.item() - sampled_focal_50.item()):.6f})")


if __name__ == "__main__":
    print("Memory Profiling Test for Point-Sampled Mask Losses (REAL DATA)")
    print("=" * 70)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device += ":" + str(torch.cuda.current_device())

    debug_memory_issue(device)

    sys.exit()
    # First verify that memory profiling is working
    test_memory_profiling_works(device)
    
    # Test large tensor memory
    test_large_tensor_memory(device)
    
    # Test loss function comparison
    test_loss_function_comparison(device)
    
    # Test accuracy first
    test_accuracy_comparison(device)
    
    # Test memory and time performance
    results = test_full_vs_sampled_losses(device)
    
    # Print summary
    print_summary(results)
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("The point-sampled mask losses should show significant memory savings")
    print("and speedup compared to full mask loss computation, especially for")
    print("large numbers of hits per event. The accuracy should remain reasonable with")
    print("20-50 sampled points per object.")
    print("\nKey benefits:")
    print("- Reduced memory usage for large events")
    print("- Faster computation time")
    print("- Maintained accuracy with sufficient sampling")
    print("- Better scalability for production use")