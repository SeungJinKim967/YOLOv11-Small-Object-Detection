#!/usr/bin/env python3
"""
YOLOv11 Small Object Detection Training Script
Performance Analysis of YOLOv11 Architecture Variants for Small Object Detection

Author: Seung Jin Kim
Lab: Computer Vision & AI Lab
Paper: Published in IEEE Access

This script implements comprehensive training pipeline for YOLOv11 variants
with focus on small object detection performance optimization.
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from ultralytics import YOLO
import wandb
from utils import setup_logging, save_results, calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv11Trainer:
    """
    YOLOv11 Training Pipeline for Small Object Detection
    
    Supports all YOLOv11 variants: n, s, m, l, x
    Optimized for objects smaller than 32x32 pixels
    """
    
    def __init__(self, model_size='yolov11n', config_path='config.yaml'):
        """
        Initialize YOLOv11 trainer
        
        Args:
            model_size (str): YOLOv11 variant ('yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x')
            config_path (str): Path to configuration file
        """
        self.model_size = model_size
        self.config_path = config_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model parameters based on research findings
        self.model_configs = {
            'yolov11n': {'params': '2.6M', 'flops': '6.5G', 'speed': 142.5},
            'yolov11s': {'params': '9.4M', 'flops': '21.5G', 'speed': 98.3},
            'yolov11m': {'params': '20.1M', 'flops': '68.2G', 'speed': 67.4},
            'yolov11l': {'params': '25.3M', 'flops': '86.9G', 'speed': 45.1},
            'yolov11x': {'params': '56.9M', 'flops': '194.9G', 'speed': 28.7}
        }
        
        logger.info(f"Initializing {model_size} trainer on {self.device}")
        logger.info(f"Model specs: {self.model_configs[model_size]}")
    
    def setup_model(self):
        """Setup YOLOv11 model with optimized configurations"""
        try:
            # Load pre-trained model
            model_path = f"models/{self.model_size}.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"Loaded pre-trained {self.model_size} model")
            else:
                # Download from ultralytics hub
                self.model = YOLO(f"{self.model_size}.pt")
                logger.info(f"Downloaded {self.model_size} model from hub")
            
            # Configure model for small object detection
            self.configure_small_object_detection()
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def configure_small_object_detection(self):
        """
        Configure model architecture for optimal small object detection
        Based on research findings from IEEE Access paper
        """
        # Small object optimization parameters
        small_object_config = {
            'mosaic': 0.5,           # Reduced mosaic for small objects
            'mixup': 0.1,            # Light mixup augmentation
            'copy_paste': 0.1,       # Copy-paste augmentation
            'degrees': 5.0,          # Reduced rotation
            'translate': 0.1,        # Reduced translation
            'scale': 0.2,            # Reduced scaling
            'fliplr': 0.5,           # Horizontal flip
            'flipud': 0.1,           # Vertical flip
            'hsv_h': 0.015,          # Hue augmentation
            'hsv_s': 0.7,            # Saturation augmentation
            'hsv_v': 0.4             # Value augmentation
        }
        
        # Apply configuration to model
        for key, value in small_object_config.items():
            if hasattr(self.model.model, key):
                setattr(self.model.model, key, value)
        
        logger.info("Applied small object detection optimizations")
    
    def train(self, data_path, epochs=100, batch_size=None, save_dir='runs/train'):
        """
        Train YOLOv11 model
        
        Args:
            data_path (str): Path to dataset YAML file
            epochs (int): Number of training epochs
            batch_size (int): Batch size (auto-determined if None)
            save_dir (str): Directory to save results
        """
        try:
            # Auto-determine batch size based on model size and GPU memory
            if batch_size is None:
                batch_size = self.get_optimal_batch_size()
            
            # Training parameters optimized for small object detection
            train_args = {
                'data': data_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': self.device,
                'project': save_dir,
                'name': f'{self.model_size}_small_objects',
                'save': True,
                'save_period': 10,
                'cache': True,
                'workers': 8,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 2.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'plots': True,
                'verbose': True
            }
            
            logger.info(f"Starting training with batch_size={batch_size}")
            logger.info(f"Training arguments: {train_args}")
            
            # Initialize Weights & Biases logging
            if self.config.get('use_wandb', False):
                wandb.init(
                    project="yolov11-small-object-detection",
                    name=f"{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=train_args
                )
            
            # Start training
            results = self.model.train(**train_args)
            
            logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def get_optimal_batch_size(self):
        """
        Determine optimal batch size based on model size and available GPU memory
        Research-based batch sizes for different model variants
        """
        if not torch.cuda.is_available():
            return 4  # CPU fallback
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # Optimal batch sizes based on research findings
        batch_size_map = {
            'yolov11n': min(32, int(gpu_memory * 4)),
            'yolov11s': min(24, int(gpu_memory * 3)),
            'yolov11m': min(16, int(gpu_memory * 2)),
            'yolov11l': min(12, int(gpu_memory * 1.5)),
            'yolov11x': min(8, int(gpu_memory))
        }
        
        optimal_batch = batch_size_map.get(self.model_size, 8)
        logger.info(f"GPU Memory: {gpu_memory:.1f}GB, Optimal batch size: {optimal_batch}")
        
        return optimal_batch
    
    def validate(self, data_path, save_dir='runs/val'):
        """
        Validate trained model
        
        Args:
            data_path (str): Path to validation dataset
            save_dir (str): Directory to save validation results
        """
        try:
            val_args = {
                'data': data_path,
                'imgsz': 640,
                'batch': 1,
                'conf': 0.001,
                'iou': 0.6,
                'device': self.device,
                'project': save_dir,
                'name': f'{self.model_size}_validation',
                'save_json': True,
                'save_hybrid': False,
                'plots': True,
                'verbose': True
            }
            
            logger.info("Starting validation...")
            results = self.model.val(**val_args)
            
            # Calculate additional metrics for small objects
            metrics = self.calculate_small_object_metrics(results)
            
            logger.info("Validation completed successfully")
            return results, metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def calculate_small_object_metrics(self, results):
        """
        Calculate specific metrics for small objects (<32x32 pixels)
        As defined in the IEEE Access paper
        """
        # Small object specific metrics
        small_obj_metrics = {
            'small_object_map50': 0.0,
            'small_object_map50_95': 0.0,
            'small_object_precision': 0.0,
            'small_object_recall': 0.0,
            'small_object_count': 0
        }
        
        # This would be implemented with actual detection results
        # For demonstration, using placeholder values
        logger.info("Calculating small object specific metrics...")
        
        return small_obj_metrics

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv11 Small Object Detection Training')
    
    parser.add_argument('--model', type=str, default='yolov11n',
                       choices=['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
                       help='YOLOv11 model variant')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto-determined if not specified)')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--save-dir', type=str, default='runs/train',
                       help='Directory to save training results')
    
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after training')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto, cpu, cuda)')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    logger.info("="*60)
    logger.info("YOLOv11 Small Object Detection Training")
    logger.info("IEEE Access Research Implementation")
    logger.info("="*60)
    
    try:
        # Initialize trainer
        trainer = YOLOv11Trainer(model_size=args.model, config_path=args.config)
        
        # Setup model
        trainer.setup_model()
        
        # Start training
        train_results = trainer.train(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.save_dir
        )
        
        # Run validation if requested
        if args.validate:
            val_results, metrics = trainer.validate(data_path=args.data)
            
            # Save comprehensive results
            save_results({
                'model': args.model,
                'train_results': train_results,
                'val_results': val_results,
                'small_object_metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
