"""
Training Orchestrator - Backend logic for managing ML training jobs

This module handles:
- State management (tracking which models are trained)
- Running training scripts with proper arguments
- Dependency management (features before classifiers)
- Error handling and retry logic
- Progress tracking

Author: Auto-generated training orchestrator
Date: 2025-11-06
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class TrainingOrchestrator:
    """Manages training jobs and tracks their completion status"""

    # Model configurations
    FEATURE_EXTRACTORS = ['plant_pretrained_base', 'imagenet_small', 'imagenet_base', 'imagenet_large']

    CLASSIFIERS = ['svm', 'random_forest', 'linear_probe']

    FINETUNE_MODELS = ['plant_pretrained_base', 'imagenet_small', 'imagenet_base', 'imagenet_large']

    def __init__(self, state_file: str = "training_state.json"):
        """Initialize orchestrator with state file path"""
        self.state_file = Path(state_file)
        self.project_root = Path(__file__).parent
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load training state from JSON file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize new state
            return self._initialize_state()

    def _initialize_state(self) -> Dict:
        """Create initial state structure"""
        state = {
            'last_updated': datetime.now().isoformat(),
            'approach_a': {
                'features': {},
                'models': {}
            },
            'approach_b': {
                'models': {}
            }
        }

        # Initialize Approach A feature extraction status
        for extractor in self.FEATURE_EXTRACTORS:
            state['approach_a']['features'][extractor] = {
                'status': 'pending',  # pending, in_progress, completed, failed
                'last_run': None,
                'error': None
            }

        # Initialize Approach A classifier status
        for extractor in self.FEATURE_EXTRACTORS:
            for classifier in self.CLASSIFIERS:
                model_id = f"{extractor}_{classifier}"
                state['approach_a']['models'][model_id] = {
                    'status': 'pending',
                    'last_run': None,
                    'error': None,
                    'metrics': {}
                }

        # Initialize Approach B fine-tuning status
        for model in self.FINETUNE_MODELS:
            state['approach_b']['models'][model] = {
                'status': 'pending',
                'last_run': None,
                'error': None,
                'metrics': {}
            }

        self._save_state(state)
        return state

    def _save_state(self, state: Optional[Dict] = None):
        """Save current state to JSON file"""
        if state is None:
            state = self.state
        state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state, indent=2, fp=f)

    def get_status_summary(self) -> Dict:
        """Get summary of training status"""
        summary = {
            'approach_a': {
                'features': {'pending': 0, 'in_progress': 0, 'completed': 0, 'failed': 0},
                'models': {'pending': 0, 'in_progress': 0, 'completed': 0, 'failed': 0}
            },
            'approach_b': {
                'models': {'pending': 0, 'in_progress': 0, 'completed': 0, 'failed': 0}
            }
        }

        # Count feature extraction status
        for feature_data in self.state['approach_a']['features'].values():
            status = feature_data['status']
            summary['approach_a']['features'][status] += 1

        # Count Approach A model status
        for model_data in self.state['approach_a']['models'].values():
            status = model_data['status']
            summary['approach_a']['models'][status] += 1

        # Count Approach B model status
        for model_data in self.state['approach_b']['models'].values():
            status = model_data['status']
            summary['approach_b']['models'][status] += 1

        return summary

    def extract_features(self, extractor: str, force: bool = False) -> bool:
        """
        Extract features using specified DINOv2 model

        Args:
            extractor: One of FEATURE_EXTRACTORS (plant_pretrained_base, imagenet_small, imagenet_base, imagenet_large)
            force: Re-extract even if already completed

        Returns:
            True if successful, False otherwise
        """
        if extractor not in self.FEATURE_EXTRACTORS:
            print(f"❌ Invalid extractor: {extractor}")
            return False

        # Check if already completed
        feature_state = self.state['approach_a']['features'][extractor]
        if feature_state['status'] == 'completed' and not force:
            print(f"✓ Features already extracted for {extractor}")
            return True

        print(f"\n🔄 Extracting features using {extractor}...")

        # Update status to in_progress
        feature_state['status'] = 'in_progress'
        feature_state['last_run'] = datetime.now().isoformat()
        self._save_state()

        # Prepare command
        script_path = self.project_root / "Approach_A_Feature_Extraction" / "extract_features.py"

        cmd = [
            sys.executable,
            str(script_path),
            '--model_type', extractor  # Use model_type argument with extractor name directly
        ]

        try:
            # Run extraction with real-time output so user can see progress
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=False,  # Show output in real-time
                text=True,
                env=env
            )

            if result.returncode == 0:
                feature_state['status'] = 'completed'
                feature_state['error'] = None
                self._save_state()
                print(f"\n✅ Features extracted successfully for {extractor}")
                return True
            else:
                feature_state['status'] = 'failed'
                feature_state['error'] = "Feature extraction failed - check output above"
                self._save_state()
                print(f"\n❌ Feature extraction failed for {extractor}")
                return False

        except Exception as e:
            feature_state['status'] = 'failed'
            feature_state['error'] = str(e)
            self._save_state()
            print(f"❌ Exception during feature extraction: {e}")
            return False

    def train_classifier(self, extractor: str, classifier: str, force: bool = False) -> bool:
        """
        Train a classifier on extracted features

        Args:
            extractor: Feature extractor used
            classifier: Classifier type (svm, random_forest, linear_probe)
            force: Retrain even if already completed

        Returns:
            True if successful, False otherwise
        """
        model_id = f"{extractor}_{classifier}"

        # Check if classifier is valid
        if classifier not in self.CLASSIFIERS:
            print(f"❌ Invalid classifier: {classifier}")
            return False

        # Check if features are extracted
        feature_state = self.state['approach_a']['features'][extractor]
        if feature_state['status'] != 'completed':
            print(f"⚠️  Features not extracted for {extractor}. Extracting now...")
            if not self.extract_features(extractor):
                print(f"❌ Cannot train {model_id} without features")
                return False

        # Check if already completed
        model_state = self.state['approach_a']['models'][model_id]
        if model_state['status'] == 'completed' and not force:
            print(f"✓ Model {model_id} already trained")
            return True

        print(f"\n🔄 Training {classifier} on {extractor} features...")

        # Update status
        model_state['status'] = 'in_progress'
        model_state['last_run'] = datetime.now().isoformat()
        self._save_state()

        # Prepare command
        script_map = {
            'svm': 'train_svm.py',
            'random_forest': 'train_random_forest.py',
            'linear_probe': 'train_linear_probe.py'
        }

        script_path = self.project_root / "Approach_A_Feature_Extraction" / script_map[classifier]

        # Construct paths for features and output
        features_dir = self.project_root / "Approach_A_Feature_Extraction" / "features" / extractor
        output_dir = self.project_root / "Approach_A_Feature_Extraction" / "results" / f"{classifier}_{extractor}"

        cmd = [
            sys.executable,
            str(script_path),
            '--features_dir', str(features_dir),
            '--output_dir', str(output_dir)
        ]

        try:
            # Run training with UTF-8 encoding to handle emojis
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )

            if result.returncode == 0:
                model_state['status'] = 'completed'
                model_state['error'] = None
                self._save_state()
                print(f"✅ Model {model_id} trained successfully")
                return True
            else:
                model_state['status'] = 'failed'
                model_state['error'] = result.stderr[-500:]
                self._save_state()
                print(f"❌ Training failed for {model_id}")
                print(f"Error: {result.stderr[-200:]}")
                return False

        except Exception as e:
            model_state['status'] = 'failed'
            model_state['error'] = str(e)
            self._save_state()
            print(f"❌ Exception during training: {e}")
            return False

    def finetune_model(self, model_type: str, epochs: int = 60, force: bool = False) -> bool:
        """
        Fine-tune a DINOv2 model (Approach B)

        Args:
            model_type: One of FINETUNE_MODELS
            epochs: Number of training epochs
            force: Retrain even if already completed

        Returns:
            True if successful, False otherwise
        """
        if model_type not in self.FINETUNE_MODELS:
            print(f"❌ Invalid model type: {model_type}")
            return False

        # Check if already completed
        model_state = self.state['approach_b']['models'][model_type]
        if model_state['status'] == 'completed' and not force:
            print(f"✓ Model {model_type} already fine-tuned")
            return True

        print(f"\n🔄 Fine-tuning {model_type} (this may take 2-6 hours)...")

        # Update status
        model_state['status'] = 'in_progress'
        model_state['last_run'] = datetime.now().isoformat()
        self._save_state()

        # Prepare command
        script_path = self.project_root / "Approach_B_Fine_Tuning" / "train_unified.py"

        cmd = [
            sys.executable,
            str(script_path),
            '--model_type', model_type,
            '--epochs', str(epochs)
        ]

        try:
            # Run fine-tuning with UTF-8 encoding to handle emojis
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=False,  # Show output in real-time
                text=True,
                env=env
            )

            if result.returncode == 0:
                model_state['status'] = 'completed'
                model_state['error'] = None
                self._save_state()
                print(f"✅ Model {model_type} fine-tuned successfully")
                return True
            else:
                model_state['status'] = 'failed'
                model_state['error'] = "Check console output for details"
                self._save_state()
                print(f"❌ Fine-tuning failed for {model_type}")
                return False

        except Exception as e:
            model_state['status'] = 'failed'
            model_state['error'] = str(e)
            self._save_state()
            print(f"❌ Exception during fine-tuning: {e}")
            return False

    def train_approach_a_full(self, extractors: Optional[List[str]] = None,
                             classifiers: Optional[List[str]] = None) -> Dict:
        """
        Train multiple Approach A models

        Args:
            extractors: List of extractors to use (None = all)
            classifiers: List of classifiers to train (None = all)

        Returns:
            Dictionary with success/failure counts
        """
        extractors = extractors or self.FEATURE_EXTRACTORS
        classifiers = classifiers or self.CLASSIFIERS

        results = {'successful': 0, 'failed': 0, 'skipped': 0}

        print(f"\n{'='*60}")
        print(f"🚀 APPROACH A TRAINING PIPELINE")
        print(f"{'='*60}")
        print(f"Extractors: {', '.join(extractors)}")
        print(f"Classifiers: {', '.join(classifiers)}")
        print(f"Total models: {len(extractors) * len(classifiers)}")
        print(f"{'='*60}\n")

        # Step 1: Extract features
        print("STEP 1: Feature Extraction")
        print("-" * 60)
        for extractor in extractors:
            success = self.extract_features(extractor)
            if not success:
                results['failed'] += 1

        # Step 2: Train classifiers
        print(f"\nSTEP 2: Train Classifiers")
        print("-" * 60)
        for extractor in extractors:
            for classifier in classifiers:
                success = self.train_classifier(extractor, classifier)
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1

        print(f"\n{'='*60}")
        print(f"✅ Approach A Complete!")
        print(f"Successful: {results['successful']} | Failed: {results['failed']}")
        print(f"{'='*60}\n")

        return results

    def train_approach_b_full(self, models: Optional[List[str]] = None,
                             epochs: int = 60) -> Dict:
        """
        Fine-tune multiple Approach B models

        Args:
            models: List of models to fine-tune (None = all)
            epochs: Training epochs per model

        Returns:
            Dictionary with success/failure counts
        """
        models = models or self.FINETUNE_MODELS

        results = {'successful': 0, 'failed': 0}

        print(f"\n{'='*60}")
        print(f"🚀 APPROACH B FINE-TUNING PIPELINE")
        print(f"{'='*60}")
        print(f"Models: {', '.join(models)}")
        print(f"Epochs: {epochs} per model")
        print(f"Estimated time: {len(models) * 2}-{len(models) * 6} hours")
        print(f"{'='*60}\n")

        for i, model in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Fine-tuning {model}...")
            success = self.finetune_model(model, epochs)
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        print(f"\n{'='*60}")
        print(f"✅ Approach B Complete!")
        print(f"Successful: {results['successful']} | Failed: {results['failed']}")
        print(f"{'='*60}\n")

        return results

    def reset_model_status(self, approach: str, model_id: str):
        """Reset a specific model's status to pending"""
        if approach == 'a':
            if model_id in self.state['approach_a']['models']:
                self.state['approach_a']['models'][model_id]['status'] = 'pending'
                self.state['approach_a']['models'][model_id]['error'] = None
                self._save_state()
                print(f"✅ Reset {model_id} status to pending")
        elif approach == 'b':
            if model_id in self.state['approach_b']['models']:
                self.state['approach_b']['models'][model_id]['status'] = 'pending'
                self.state['approach_b']['models'][model_id]['error'] = None
                self._save_state()
                print(f"✅ Reset {model_id} status to pending")

    def reset_all_status(self):
        """Reset all model statuses to pending"""
        self.state = self._initialize_state()
        print("✅ All model statuses reset to pending")


if __name__ == "__main__":
    # Quick test
    orchestrator = TrainingOrchestrator()
    print("Training Orchestrator initialized")
    print("\nCurrent status:")
    summary = orchestrator.get_status_summary()
    print(json.dumps(summary, indent=2))
