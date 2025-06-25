#!/usr/bin/env python3
"""
Memory profiling script to compare full mask loss vs point-sampled mask loss.
"""

import torch
import torch.nn.functional as F
import time
import psutil
import os
from memory_profiler import profile
import gc

# Import the loss functions
from hepattn.models.loss import (
    mask_dice_loss, 
    mask_focal_loss,
    point_sampled_mask_dice_loss,
    point_sampled_mask_focal_loss
)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_test_data(batch_size=2, num_objects=10, num_points_per_track=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Create test data for mask loss computation."""
    print(f"Creating test data: batch_size={batch_size}, num_objects={num_objects}, num_points_per_track={num_points_per_track}")
    
    # Create random logits and targets
    pred_logits = torch.randn(batch_size, num_objects, num_points_per_track, device=device)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_points_per_track), dtype=torch.float, device=device)
    
    # Create mask (some objects are invalid)
    mask = torch.ones(batch_size, num_objects, num_points_per_track, dtype=torch.bool, device=device)
    # Make some objects invalid (all zeros in mask)
    for b in range(batch_size):
        invalid_objects = torch.randint(0, num_objects, (num_objects // 3,))  # 1/3 invalid
        mask[b, invalid_objects, :] = False
    
    # Create weight tensor
    weight = torch.ones(batch_size, num_objects, num_points_per_track, device=device)
    weight[targets == 0] = 0.1  # Lower weight for negative samples
    
    return pred_logits, targets, mask, weight


def time_and_memory_test(func, name, *args, **kwargs):
    """Test a function and measure time and memory usage."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    
    # Clear cache and garbage collect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Measure initial memory
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    # Time the function
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Measure peak memory
    peak_memory = get_memory_usage()
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Memory increase: {peak_memory - initial_memory:.2f} MB")
    print(f"Time: {end_time - start_time:.4f} seconds")
    print(f"Result: {result.item():.6f}")
    
    # Clear cache and garbage collect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return result, end_time - start_time, peak_memory - initial_memory


def test_full_vs_sampled_losses():
    """Compare full mask losses vs point-sampled versions."""
    
    # Test parameters
    test_configs = [
        {"batch_size": 2, "num_objects": 10, "num_points_per_track": 1000, "name": "Small"},
        {"batch_size": 4, "num_objects": 20, "num_points_per_track": 2000, "name": "Medium"},
        {"batch_size": 8, "num_objects": 40, "num_points_per_track": 5000, "name": "Large"},
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = {}
    
    for config in test_configs:
        print(f"\n{'#'*60}")
        print(f"Testing {config['name']} configuration")
        print(f"{'#'*60}")
        
        # Create test data
        pred_logits, targets, mask, weight = create_test_data(
            batch_size=config['batch_size'],
            num_objects=config['num_objects'], 
            num_points_per_track=config['num_points_per_track'],
            device=device
        )
        
        config_results = {}
        
        # Test full dice loss
        result, time_taken, memory_used = time_and_memory_test(
            mask_dice_loss, 
            "Full Dice Loss",
            pred_logits, targets, mask=mask, weight=weight
        )
        config_results['full_dice'] = {'time': time_taken, 'memory': memory_used, 'result': result.item()}
        
        # Test point-sampled dice loss
        result, time_taken, memory_used = time_and_memory_test(
            point_sampled_mask_dice_loss,
            "Point-sampled Dice Loss (20 points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=20
        )
        config_results['sampled_dice_20'] = {'time': time_taken, 'memory': memory_used, 'result': result.item()}
        
        # Test point-sampled dice loss with more points
        result, time_taken, memory_used = time_and_memory_test(
            point_sampled_mask_dice_loss,
            "Point-sampled Dice Loss (50 points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=50
        )
        config_results['sampled_dice_50'] = {'time': time_taken, 'memory': memory_used, 'result': result.item()}
        
        # Test full focal loss
        result, time_taken, memory_used = time_and_memory_test(
            mask_focal_loss,
            "Full Focal Loss",
            pred_logits, targets, mask=mask, weight=weight
        )
        config_results['full_focal'] = {'time': time_taken, 'memory': memory_used, 'result': result.item()}
        
        # Test point-sampled focal loss
        result, time_taken, memory_used = time_and_memory_test(
            point_sampled_mask_focal_loss,
            "Point-sampled Focal Loss (20 points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=20
        )
        config_results['sampled_focal_20'] = {'time': time_taken, 'memory': memory_used, 'result': result.item()}
        
        # Test point-sampled focal loss with more points
        result, time_taken, memory_used = time_and_memory_test(
            point_sampled_mask_focal_loss,
            "Point-sampled Focal Loss (50 points)",
            pred_logits, targets, mask=mask, weight=weight, num_points_per_track=50
        )
        config_results['sampled_focal_50'] = {'time': time_taken, 'memory': memory_used, 'result': result.item()}
        
        results[config['name']] = config_results
    
    return results


def print_summary(results):
    """Print a summary of the results."""
    print(f"\n{'='*80}")
    print("MEMORY AND TIME COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for config_name, config_results in results.items():
        print(f"\n{config_name} Configuration:")
        print("-" * 40)
        
        # Dice loss comparison
        full_dice = config_results['full_dice']
        sampled_dice_20 = config_results['sampled_dice_20']
        sampled_dice_50 = config_results['sampled_dice_50']
        
        print(f"Dice Loss:")
        print(f"  Full:           {full_dice['memory']:6.2f} MB, {full_dice['time']:6.4f}s")
        print(f"  Sampled (20):   {sampled_dice_20['memory']:6.2f} MB, {sampled_dice_20['time']:6.4f}s")
        print(f"  Sampled (50):   {sampled_dice_50['memory']:6.2f} MB, {sampled_dice_50['time']:6.4f}s")
        print(f"  Memory savings (20): {(1 - sampled_dice_20['memory']/full_dice['memory'])*100:5.1f}%")
        print(f"  Memory savings (50): {(1 - sampled_dice_50['memory']/full_dice['memory'])*100:5.1f}%")
        print(f"  Speedup (20):   {full_dice['time']/sampled_dice_20['time']:5.1f}x")
        print(f"  Speedup (50):   {full_dice['time']/sampled_dice_50['time']:5.1f}x")
        
        # Focal loss comparison
        full_focal = config_results['full_focal']
        sampled_focal_20 = config_results['sampled_focal_20']
        sampled_focal_50 = config_results['sampled_focal_50']
        
        print(f"Focal Loss:")
        print(f"  Full:           {full_focal['memory']:6.2f} MB, {full_focal['time']:6.4f}s")
        print(f"  Sampled (20):   {sampled_focal_20['memory']:6.2f} MB, {sampled_focal_20['time']:6.4f}s")
        print(f"  Sampled (50):   {sampled_focal_50['memory']:6.2f} MB, {sampled_focal_50['time']:6.4f}s")
        print(f"  Memory savings (20): {(1 - sampled_focal_20['memory']/full_focal['memory'])*100:5.1f}%")
        print(f"  Memory savings (50): {(1 - sampled_focal_50['memory']/full_focal['memory'])*100:5.1f}%")
        print(f"  Speedup (20):   {full_focal['time']/sampled_focal_20['time']:5.1f}x")
        print(f"  Speedup (50):   {full_focal['time']/sampled_focal_50['time']:5.1f}x")


def test_accuracy_comparison():
    """Test if the point-sampled losses give similar results to full losses."""
    print(f"\n{'='*60}")
    print("ACCURACY COMPARISON")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # # Create test data
    # pred_logits, targets, mask, weight = create_test_data(
    #     batch_size=1, num_objects=500, num_points_per_track=10000, device=device
    # )
    

    print(pred_logits.shape)
    print(targets.shape)
    print(mask.shape)
    print(weight.shape)
    
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
    print("Memory Profiling Test for Point-Sampled Mask Losses")
    print("=" * 60)
    
    # Test accuracy first
    test_accuracy_comparison()
    
    # Test memory and time performance
    results = test_full_vs_sampled_losses()
    
    # Print summary
    print_summary(results)
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("The point-sampled mask losses should show significant memory savings")
    print("and speedup compared to full mask loss computation, especially for")
    print("large numbers of points. The accuracy should remain reasonable with")
    print("20-50 sampled points per object.") 




# PYTORCH_CMD="python src/hepattn/experiments/trackml/test_memory.py"
# # Pixi commnand that runs the python command inside the pixi env
# PIXI_CMD="pixi run $PYTORCH_CMD"
# # Apptainer command that runs the pixi command inside the pixi apptainer image
# APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/pduckett /share/rcifdata/pduckett/hepattn/pixi.sif $PIXI_CMD"