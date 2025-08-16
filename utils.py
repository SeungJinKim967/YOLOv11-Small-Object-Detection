#!/usr/bin/env python3
"""
YOLOv11 Small Object Detection Utility Functions
Performance Analysis of YOLOv11 Architecture Variants for Small Object Detection

Author: Seung Jin Kim
Lab: Computer Vision & AI Lab
Paper: Published in IEEE Access

This module contains utility functions for training, evaluation, and analysis
of YOLOv11 models with focus on small object detection performance.
"""

import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
import scipy.stats as stats
from scipy import interpolate
import cv2
import torch

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration for the project
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup basic configuration
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    # Configure specific loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def save_results(results: Dict, save_path: str, format: str = "json"):
    """
    Save results to file in specified format
    
    Args:
        results: Results dictionary to save
        save_path: Path to save results
        format: Save format ('json', 'yaml', 'csv')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format.lower() == "json":
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format.lower() == "yaml":
        with open(save_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    elif format.lower() == "csv":
        if isinstance(results, dict):
            pd.DataFrame([results]).to_csv(save_path, index=False)
        else:
            pd.DataFrame(results).to_csv(save_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def calculate_confidence_intervals(data: List[float], confidence: float = 0.95) -> Dict:
    """
    Calculate confidence intervals for given data
    Implementation based on IEEE Access paper methodology
    
    Args:
        data: List of numerical values
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Dictionary with confidence interval statistics
    """
    if len(data) < 2:
        return {"mean": data[0] if data else 0, "ci_lower": 0, "ci_upper": 0, "std": 0}
    
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    # Calculate confidence interval using t-distribution
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    ci_lower = mean - t_critical * se
    ci_upper = mean + t_critical * se
    
    return {
        "mean": float(mean),
        "std": float(std),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence,
        "sample_size": n
    }

def statistical_significance_test(group1: List[float], group2: List[float], 
                                test_type: str = "ttest") -> Dict:
    """
    Perform statistical significance testing between two groups
    
    Args:
        group1: First group of values
        group2: Second group of values
        test_type: Type of test ('ttest', 'mannwhitney', 'wilcoxon')
        
    Returns:
        Dictionary with test results
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if test_type.lower() == "ttest":
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Independent t-test"
    elif test_type.lower() == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(group1, group2)
        test_name = "Mann-Whitney U test"
    elif test_type.lower() == "wilcoxon":
        statistic, p_value = stats.wilcoxon(group1, group2)
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError(f"Unsupported test type: {test_type}")
    
    # Effect size calculation (Cohen's d for t-test)
    if test_type.lower() == "ttest":
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
    else:
        effect_size = None
    
    return {
        "test_name": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size": float(effect_size) if effect_size is not None else None,
        "group1_mean": float(np.mean(group1)),
        "group2_mean": float(np.mean(group2)),
        "group1_std": float(np.std(group1, ddof=1)),
        "group2_std": float(np.std(group2, ddof=1))
    }

def calculate_metrics(predictions: Dict, ground_truth: Dict) -> Dict:
    """
    Calculate comprehensive detection metrics
    
    Args:
        predictions: Prediction results
        ground_truth: Ground truth annotations
        
    Returns:
        Dictionary with calculated metrics
    """
    # Placeholder implementation
    # In real implementation, this would calculate:
    # - mAP@0.5, mAP@0.5:0.95
    # - Precision, Recall, F1-score
    # - Small object specific metrics
    
    metrics = {
        "mAP_50": 0.0,
        "mAP_50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "small_object_map": 0.0
    }
    
    return metrics

def analyze_small_objects(detections: List, image_size: Tuple[int, int], 
                         threshold: int = 32) -> Dict:
    """
    Analyze small object detection performance
    Core function for IEEE Access paper analysis
    
    Args:
        detections: List of detection results
        image_size: Image dimensions (height, width)
        threshold: Small object threshold in pixels (default: 32)
        
    Returns:
        Dictionary with small object analysis results
    """
    small_objects = []
    medium_objects = []
    large_objects = []
    
    for detection in detections:
        # Extract bounding box dimensions
        bbox = detection.get('bbox', [0, 0, 0, 0])  # [x, y, w, h]
        width, height = bbox[2], bbox[3]
        area = width * height
        
        # Classify by size
        if max(width, height) < threshold:
            small_objects.append(detection)
        elif area < (threshold * 3)**2:
            medium_objects.append(detection)
        else:
            large_objects.append(detection)
    
    total_objects = len(detections)
    
    analysis = {
        "total_objects": total_objects,
        "small_objects": {
            "count": len(small_objects),
            "percentage": len(small_objects) / total_objects * 100 if total_objects > 0 else 0,
            "avg_size": np.mean([max(obj['bbox'][2], obj['bbox'][3]) for obj in small_objects]) if small_objects else 0
        },
        "medium_objects": {
            "count": len(medium_objects),
            "percentage": len(medium_objects) / total_objects * 100 if total_objects > 0 else 0
        },
        "large_objects": {
            "count": len(large_objects),
            "percentage": len(large_objects) / total_objects * 100 if total_objects > 0 else 0
        },
        "size_distribution": {
            "small_threshold": threshold,
            "image_size": image_size
        }
    }
    
    return analysis

def create_performance_comparison_plot(results: Dict, save_path: str = None):
    """
    Create performance comparison plots for YOLOv11 variants
    
    Args:
        results: Dictionary with model results
        save_path: Optional path to save the plot
    """
    # Extract data for plotting
    models = list(results.keys())
    map_50 = [results[model].get('mAP50', 0) for model in models]
    map_50_95 = [results[model].get('mAP50_95', 0) for model in models]
    fps = [results[model].get('fps', 0) for model in models]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # mAP@0.5 comparison
    bars1 = ax1.bar(models, map_50, color='skyblue', alpha=0.8)
    ax1.set_title('mAP@0.5 Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('mAP@0.5 (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars1, map_50):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # mAP@0.5:0.95 comparison
    bars2 = ax2.bar(models, map_50_95, color='lightcoral', alpha=0.8)
    ax2.set_title('mAP@0.5:0.95 Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('mAP@0.5:0.95 (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    
    for bar, value in zip(bars2, map_50_95):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # FPS comparison
    bars3 = ax3.bar(models, fps, color='lightgreen', alpha=0.8)
    ax3.set_title('Inference Speed (FPS) Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('FPS', fontsize=12)
    
    for bar, value in zip(bars3, fps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy vs Speed scatter plot
    ax4.scatter(fps, map_50_95, s=100, alpha=0.7, c=['red', 'orange', 'yellow', 'green', 'blue'])
    for i, model in enumerate(models):
        ax4.annotate(model, (fps[i], map_50_95[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('FPS', fontsize=12)
    ax4.set_ylabel('mAP@0.5:0.95 (%)', fontsize=12)
    ax4.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_efficiency_analysis_plot(results: Dict, save_path: str = None):
    """
    Create computational efficiency analysis plots
    
    Args:
        results: Dictionary with efficiency results
        save_path: Optional path to save the plot
    """
    models = list(results.keys())
    memory = [results[model].get('memory_gb', 0) for model in models]
    energy = [results[model].get('energy_watts', 0) for model in models]
    fps = [results[model].get('fps', 0) for model in models]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Memory usage
    bars1 = ax1.bar(models, memory, color='purple', alpha=0.7)
    ax1.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Memory (GB)', fontsize=12)
    
    for bar, value in zip(bars1, memory):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}GB', ha='center', va='bottom', fontweight='bold')
    
    # Energy consumption
    bars2 = ax2.bar(models, energy, color='red', alpha=0.7)
    ax2.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Energy (Watts)', fontsize=12)
    
    for bar, value in zip(bars2, energy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{value:.1f}W', ha='center', va='bottom', fontweight='bold')
    
    # Efficiency scatter (FPS vs Energy)
    ax3.scatter(energy, fps, s=100, alpha=0.7, c=['red', 'orange', 'yellow', 'green', 'blue'])
    for i, model in enumerate(models):
        ax3.annotate(model, (energy[i], fps[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Energy (Watts)', fontsize=12)
    ax3.set_ylabel('FPS', fontsize=12)
    ax3.set_title('Energy Efficiency Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_small_object_analysis_plot(small_obj_results: Dict, save_path: str = None):
    """
    Create small object detection analysis plots
    Key visualization for IEEE Access paper
    
    Args:
        small_obj_results: Small object analysis results
        save_path: Optional path to save the plot
    """
    models = list(small_obj_results.keys())
    small_map = [small_obj_results[model].get('small_object_map', 0) for model in models]
    detection_rate = [small_obj_results[model].get('detection_rate', 0) for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Small object mAP
    bars1 = ax1.bar(models, small_map, color='teal', alpha=0.8)
    ax1.set_title('Small Object Detection Performance\n(Objects <32×32 pixels)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Small Object mAP (%)', fontsize=12)
    ax1.set_ylim(0, max(small_map) * 1.2 if small_map else 100)
    
    for bar, value in zip(bars1, small_map):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Detection rate comparison
    x = np.arange(len(models))
    width = 0.35
    
    ax2.bar(x, small_map, width, label='Small Object mAP', alpha=0.8, color='teal')
    ax2.bar(x + width, detection_rate, width, label='Detection Rate', alpha=0.8, color='orange')
    
    ax2.set_xlabel('YOLOv11 Variants', fontsize=12)
    ax2.set_ylabel('Performance (%)', fontsize=12)
    ax2.set_title('Small Object Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def export_results_to_excel(results: Dict, save_path: str):
    """
    Export comprehensive results to Excel file
    
    Args:
        results: Complete results dictionary
        save_path: Path to save Excel file
    """
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        # Performance metrics
        if 'performance' in results:
            df_performance = pd.DataFrame(results['performance']).T
            df_performance.to_excel(writer, sheet_name='Performance_Metrics')
        
        # Efficiency metrics
        if 'efficiency' in results:
            df_efficiency = pd.DataFrame(results['efficiency']).T
            df_efficiency.to_excel(writer, sheet_name='Efficiency_Metrics')
        
        # Small object analysis
        if 'small_objects' in results:
            df_small_obj = pd.DataFrame(results['small_objects']).T
            df_small_obj.to_excel(writer, sheet_name='Small_Object_Analysis')
        
        # Statistical analysis
        if 'statistics' in results:
            df_stats = pd.DataFrame(results['statistics']).T
            df_stats.to_excel(writer, sheet_name='Statistical_Analysis')

def validate_model_config(config: Dict) -> bool:
    """
    Validate model configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Boolean indicating if configuration is valid
    """
    required_keys = ['models', 'datasets', 'training', 'evaluation']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate model configurations
    supported_models = ['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x']
    for model in config['models']:
        if model not in supported_models:
            raise ValueError(f"Unsupported model: {model}")
    
    return True

def get_model_info(model_name: str) -> Dict:
    """
    Get model information and specifications
    
    Args:
        model_name: Name of the YOLOv11 model
        
    Returns:
        Dictionary with model specifications
    """
    model_specs = {
        'yolov11n': {
            'parameters': '2.6M',
            'flops': '6.5G',
            'size_mb': 5.9,
            'description': 'Nano - fastest inference, lowest accuracy'
        },
        'yolov11s': {
            'parameters': '9.4M',
            'flops': '21.5G', 
            'size_mb': 21.5,
            'description': 'Small - balanced speed and accuracy'
        },
        'yolov11m': {
            'parameters': '20.1M',
            'flops': '68.2G',
            'size_mb': 49.7,
            'description': 'Medium - optimal trade-off'
        },
        'yolov11l': {
            'parameters': '25.3M',
            'flops': '86.9G',
            'size_mb': 64.8,
            'description': 'Large - high accuracy'
        },
        'yolov11x': {
            'parameters': '56.9M',
            'flops': '194.9G',
            'size_mb': 143.7,
            'description': 'Extra Large - maximum accuracy'
        }
    }
    
    return model_specs.get(model_name, {})

def print_system_info():
    """Print system information for reproducibility"""
    import platform
    import sys
    
    print("="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA Available: No")
    
    print("="*50)

def format_results_for_paper(results: Dict) -> str:
    """
    Format results for academic paper presentation
    
    Args:
        results: Results dictionary
        
    Returns:
        Formatted string suitable for paper
    """
    formatted = []
    formatted.append("Performance Analysis Results")
    formatted.append("="*40)
    
    for model, metrics in results.items():
        formatted.append(f"\n{model.upper()}:")
        formatted.append(f"  mAP@0.5: {metrics.get('mAP50', 0):.1f}%")
        formatted.append(f"  mAP@0.5:0.95: {metrics.get('mAP50_95', 0):.1f}%")
        formatted.append(f"  FPS: {metrics.get('fps', 0):.1f}")
        formatted.append(f"  Memory: {metrics.get('memory_gb', 0):.1f} GB")
        formatted.append(f"  Energy: {metrics.get('energy_watts', 0):.1f} W")
    
    return "\n".join(formatted)

# Initialize logging when module is imported
setup_logging()
