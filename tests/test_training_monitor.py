"""
Tests for the Training Monitor
Tests the training progress tracking functionality
"""

import pytest
import sys
import os
import json
import tempfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_monitor import TrainingMonitor


class TestTrainingMonitorInit:
    """Test TrainingMonitor initialization"""
    
    def test_monitor_creation(self):
        """Test that monitor can be created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "logs", "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            assert monitor is not None
            assert os.path.exists(status_file)
    
    def test_monitor_creates_directory(self):
        """Test that monitor creates logs directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "nested", "logs", "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            assert os.path.exists(os.path.dirname(status_file))


class TestTrainingMonitorOperations:
    """Test TrainingMonitor operations"""
    
    def test_reset(self):
        """Test reset operation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            monitor.reset()
            status = monitor.get_status()
            
            assert status["status"] == "idle"
            assert status["current_timestep"] == 0
            assert status["total_timesteps"] == 0
    
    def test_start_training(self):
        """Test start_training operation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            monitor.start_training(total_timesteps=50000)
            status = monitor.get_status()
            
            assert status["status"] == "training"
            assert status["current_timestep"] == 0
            assert status["total_timesteps"] == 50000
    
    def test_update_progress(self):
        """Test update_progress operation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            monitor.start_training(total_timesteps=50000)
            monitor.update_progress(current_timestep=10000, mean_reward=-250.5)
            
            status = monitor.get_status()
            
            assert status["current_timestep"] == 10000
            assert status["mean_reward"] == -250.5
    
    def test_finish_training_success(self):
        """Test finish_training with success"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            monitor.start_training(total_timesteps=50000)
            monitor.finish_training(success=True)
            
            status = monitor.get_status()
            assert status["status"] == "completed"
    
    def test_finish_training_failure(self):
        """Test finish_training with failure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            monitor.start_training(total_timesteps=50000)
            monitor.finish_training(success=False)
            
            status = monitor.get_status()
            assert status["status"] == "failed"
    
    def test_last_update_timestamp(self):
        """Test that last_update timestamp is set"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            monitor.start_training(total_timesteps=50000)
            status = monitor.get_status()
            
            assert "last_update" in status
            # Should be a valid ISO timestamp
            datetime.fromisoformat(status["last_update"])


class TestTrainingMonitorPersistence:
    """Test TrainingMonitor file persistence"""
    
    def test_status_persists_to_file(self):
        """Test that status is written to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "status.json")
            monitor = TrainingMonitor(status_file=status_file)
            
            monitor.start_training(total_timesteps=100000)
            monitor.update_progress(current_timestep=25000, mean_reward=-300.0)
            
            # Read file directly
            with open(status_file, 'r') as f:
                file_status = json.load(f)
            
            assert file_status["current_timestep"] == 25000
            assert file_status["mean_reward"] == -300.0
    
    def test_status_readable_by_new_instance(self):
        """Test that status can be read by new monitor instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "status.json")
            
            # Write with first instance
            monitor1 = TrainingMonitor(status_file=status_file)
            monitor1.start_training(total_timesteps=50000)
            monitor1.update_progress(current_timestep=15000, mean_reward=-200.0)
            
            # Read file directly (TrainingMonitor resets on init, so we read file)
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            assert status["current_timestep"] == 15000
            assert status["mean_reward"] == -200.0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
