"""
Training Manager - Interactive CLI for ML Model Training

Provides an easy-to-use terminal interface for:
- Training Approach A models (feature extraction + classifiers)
- Training Approach B models (fine-tuning)
- Individual model selection
- Batch training
- Status monitoring
- Resume capability

Author: Auto-generated training manager
Date: 2025-11-06
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from training_orchestrator import TrainingOrchestrator


class TrainingManager:
    """Interactive CLI for managing model training"""

    def __init__(self):
        self.orchestrator = TrainingOrchestrator()
        self.running = True

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70 + "\n")

    def print_status_bar(self):
        """Print current training status"""
        summary = self.orchestrator.get_status_summary()

        print("📊 TRAINING STATUS")
        print("-" * 70)

        # Approach A Features
        a_features = summary['approach_a']['features']
        print(f"Approach A - Features:     ", end="")
        print(f"✅ {a_features['completed']}/4  ", end="")
        print(f"🔄 {a_features['in_progress']}  ", end="")
        print(f"❌ {a_features['failed']}  ", end="")
        print(f"⏳ {a_features['pending']}")

        # Approach A Models
        a_models = summary['approach_a']['models']
        print(f"Approach A - Classifiers:  ", end="")
        print(f"✅ {a_models['completed']}/12  ", end="")
        print(f"🔄 {a_models['in_progress']}  ", end="")
        print(f"❌ {a_models['failed']}  ", end="")
        print(f"⏳ {a_models['pending']}")

        # Approach B Models
        b_models = summary['approach_b']['models']
        print(f"Approach B - Fine-tuned:   ", end="")
        print(f"✅ {b_models['completed']}/4  ", end="")
        print(f"🔄 {b_models['in_progress']}  ", end="")
        print(f"❌ {b_models['failed']}  ", end="")
        print(f"⏳ {b_models['pending']}")

        print("-" * 70 + "\n")

    def main_menu(self):
        """Display main menu and handle selection"""
        self.clear_screen()
        self.print_header("🌱 PLANT IDENTIFICATION - TRAINING MANAGER")
        self.print_status_bar()

        print("MAIN MENU:")
        print()
        print("EXPERIMENT A: MIXED DATASET (Herb + Field)")
        print(f"  1. Approach A - Feature Extraction & Classifiers (Train + Validate [Both mixed with Herbarium and Photo Imgaes (207 field images)])")
        print(f"  2. Approach B - Fine-Tuning (Train + Validate [Both mixed with Herbarium and Photo Imgaes (207 field images)])")
        print("  3. Run Full Pipeline (All 16 Models)")
        print("  4. View Detailed Status")
        print("  5. Generate Comparison Report")
        print("  6. Generate Visualizations")
        print("  7. Reset Model Status")
        print()
        print("EXPERIMENT B: CROSS-DOMAIN GENERALIZATION (Herbarium → Field)")
        print("  8. Cross-Domain Workflow (CROSS-DOMAIN GENERALIZATION (Herbarium → Field))")
        print("  0. Exit")
        print()

        choice = input("Select option (0-8): ").strip()

        if choice == '1':
            self.approach_a_menu()
        elif choice == '2':
            self.approach_b_menu()
        elif choice == '3':
            self.run_full_pipeline()
        elif choice == '4':
            self.view_detailed_status()
        elif choice == '5':
            self.generate_reports()
        elif choice == '6':
            self.visualization_menu()
        elif choice == '7':
            self.reset_status_menu()
        elif choice == '8':
            self.crossdomain_menu()
        elif choice == '0':
            self.running = False
            print("\n👋 Goodbye!\n")
        else:
            input("\n❌ Invalid option. Press Enter to continue...")

    def approach_a_menu(self):
        """Approach A submenu"""
        self.clear_screen()
        self.print_header("APPROACH A - Feature Extraction & Classifiers")

        print("OPTIONS:")
        print("  1. Extract Features (Select Extractor)")
        print("  2. Train Classifier (Select Model)")
        print("  3. Run All Approach A (12 Models)")
        print("  4. Custom Batch (Select Multiple)")
        print("  0. Back to Main Menu")
        print()

        choice = input("Select option (0-4): ").strip()

        if choice == '1':
            self.extract_features_menu()
        elif choice == '2':
            self.train_classifier_menu()
        elif choice == '3':
            self.run_all_approach_a()
        elif choice == '4':
            self.custom_approach_a_batch()
        elif choice == '0':
            return
        else:
            input("\n❌ Invalid option. Press Enter to continue...")
            self.approach_a_menu()

    def extract_features_menu(self):
        """Feature extraction submenu"""
        self.clear_screen()
        self.print_header("EXTRACT FEATURES")

        extractors = self.orchestrator.FEATURE_EXTRACTORS

        print("Select Feature Extractor:")
        for i, extractor in enumerate(extractors, 1):
            status = self.orchestrator.state['approach_a']['features'][extractor]['status']
            icon = '✅' if status == 'completed' else '⏳' if status == 'pending' else '🔄' if status == 'in_progress' else '❌'
            print(f"  {i}. {icon} {extractor}")
        print("  0. Back")
        print()

        choice = input("Select extractor (0-4): ").strip()

        if choice == '0':
            self.approach_a_menu()
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(extractors):
                extractor = extractors[idx]
                print(f"\n🔄 Starting feature extraction for {extractor}...")
                print("This may take 15-30 minutes.\n")

                force = False
                status = self.orchestrator.state['approach_a']['features'][extractor]['status']
                if status == 'completed':
                    rerun = input("Features already extracted. Re-extract? (y/N): ").strip().lower()
                    force = (rerun == 'y')
                    if not force:
                        input("\nSkipping. Press Enter to continue...")
                        self.approach_a_menu()
                        return

                success = self.orchestrator.extract_features(extractor, force=force)

                if success:
                    input("\n✅ Feature extraction complete! Press Enter to continue...")
                else:
                    input("\n❌ Feature extraction failed. Press Enter to continue...")

                self.approach_a_menu()
            else:
                input("\n❌ Invalid selection. Press Enter to continue...")
                self.extract_features_menu()
        except ValueError:
            input("\n❌ Invalid input. Press Enter to continue...")
            self.extract_features_menu()

    def train_classifier_menu(self):
        """Classifier training submenu"""
        self.clear_screen()
        self.print_header("TRAIN CLASSIFIER")

        # Step 1: Select extractor
        extractors = self.orchestrator.FEATURE_EXTRACTORS
        print("Step 1: Select Feature Extractor")
        for i, extractor in enumerate(extractors, 1):
            feature_status = self.orchestrator.state['approach_a']['features'][extractor]['status']
            icon = '✅' if feature_status == 'completed' else '⚠️ '
            print(f"  {i}. {icon} {extractor}")
        print("  0. Back")
        print()

        extractor_choice = input("Select extractor (0-4): ").strip()

        if extractor_choice == '0':
            self.approach_a_menu()
            return

        try:
            extractor_idx = int(extractor_choice) - 1
            if not (0 <= extractor_idx < len(extractors)):
                input("\n❌ Invalid selection. Press Enter to continue...")
                self.train_classifier_menu()
                return

            extractor = extractors[extractor_idx]

            # Step 2: Select classifier
            print(f"\nStep 2: Select Classifier for {extractor}")
            classifiers = self.orchestrator.CLASSIFIERS
            for i, classifier in enumerate(classifiers, 1):
                model_id = f"{extractor}_{classifier}"
                status = self.orchestrator.state['approach_a']['models'][model_id]['status']
                icon = '✅' if status == 'completed' else '⏳' if status == 'pending' else '🔄' if status == 'in_progress' else '❌'
                print(f"  {i}. {icon} {classifier}")
            print("  4. Train All Three")
            print("  0. Back")
            print()

            classifier_choice = input("Select classifier (0-4): ").strip()

            if classifier_choice == '0':
                self.train_classifier_menu()
                return

            force = False

            # Step 3: Configure n_jobs for SVM/Random Forest
            n_jobs = -1  # Default
            if classifier_choice == '4':
                # Training all three - ask for n_jobs once
                print("\nSome classifiers (SVM, Random Forest) support parallel processing.")
                n_jobs = self._get_n_jobs_from_user()
            else:
                # Single classifier - check if it needs n_jobs
                try:
                    classifier_idx = int(classifier_choice) - 1
                    if 0 <= classifier_idx < len(classifiers):
                        classifier = classifiers[classifier_idx]
                        if classifier in ['svm', 'random_forest']:
                            n_jobs = self._get_n_jobs_from_user()
                except ValueError:
                    pass  # Will be caught later

            if classifier_choice == '4':
                # Train all three
                print(f"\n🔄 Training all classifiers for {extractor}...")

                # Check if any models are already completed
                completed_models = []
                for classifier in classifiers:
                    model_id = f"{extractor}_{classifier}"
                    status = self.orchestrator.state['approach_a']['models'][model_id]['status']
                    if status == 'completed':
                        completed_models.append(classifier)

                # Ask about retraining if any are completed
                if completed_models:
                    print(f"\n⚠️  Found {len(completed_models)} already trained model(s): {', '.join(completed_models)}")
                    retrain_all = input("Retrain already completed models? (y/N): ").strip().lower()
                    force = (retrain_all == 'y')

                for classifier in classifiers:
                    model_id = f"{extractor}_{classifier}"
                    status = self.orchestrator.state['approach_a']['models'][model_id]['status']
                    if status == 'completed' and not force:
                        print(f"\n⏭️  Skipping {classifier} (already completed)")
                        continue
                    print(f"\n{'='*60}")
                    print(f"Training {classifier}...")
                    print('='*60)
                    self.orchestrator.train_classifier(extractor, classifier, force=force, n_jobs=n_jobs)

                input("\n✅ All classifiers trained! Press Enter to continue...")
                self.approach_a_menu()
            else:
                classifier_idx = int(classifier_choice) - 1
                if 0 <= classifier_idx < len(classifiers):
                    classifier = classifiers[classifier_idx]
                    model_id = f"{extractor}_{classifier}"
                    status = self.orchestrator.state['approach_a']['models'][model_id]['status']

                    if status == 'completed':
                        rerun = input(f"\nModel already trained. Retrain? (y/N): ").strip().lower()
                        force = (rerun == 'y')
                        if not force:
                            input("\nSkipping. Press Enter to continue...")
                            self.approach_a_menu()
                            return

                    print(f"\n🔄 Training {classifier} on {extractor} features...")
                    success = self.orchestrator.train_classifier(extractor, classifier, force=force, n_jobs=n_jobs)

                    if success:
                        input("\n✅ Training complete! Press Enter to continue...")
                    else:
                        input("\n❌ Training failed. Press Enter to continue...")

                    self.approach_a_menu()
                else:
                    input("\n❌ Invalid selection. Press Enter to continue...")
                    self.train_classifier_menu()

        except ValueError:
            input("\n❌ Invalid input. Press Enter to continue...")
            self.train_classifier_menu()

    def _get_n_jobs_from_user(self) -> int:
        """
        Interactively ask user for n_jobs parameter.
        On Windows, defaults to 1 core due to Python 3.13 multiprocessing limitations.

        Returns:
            int: Number of parallel jobs (1 for sequential, -1 for all CPUs)
        """
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        is_windows = sys.platform == 'win32'

        print(f"\n⚙️  Configure Parallel Processing")
        print(f"   Your CPU has {cpu_count} cores")

        # Show Windows warning if applicable
        if is_windows:
            print(f"   ⚠️  Note: Python 3.13 on Windows has limited multiprocessing support")
            print(f"   ⚠️  Parallel processing may fail - use option 1 for guaranteed stability")

        print(f"   Options:")
        if is_windows:
            print(f"      [1] Use 1 core (RECOMMENDED - 100% stable)")
            print(f"      [2] Use {cpu_count // 2} cores (experimental, may fallback to 1 core)")
            print(f"      [3] Use ALL {cpu_count} cores (experimental, may fallback to 1 core)")
        else:
            print(f"      [1] Use 1 core (slowest, most stable)")
            print(f"      [2] Use {cpu_count // 2} cores (balanced)")
            print(f"      [3] Use ALL {cpu_count} cores (fastest)")
        print(f"      [4] Custom (enter specific number)")
        print()

        # Default to 1 on Windows, all cores on other platforms
        default_option = '1' if is_windows else '3'
        default_cores = 1 if is_windows else -1

        choice = input(f"Select option (1-4) [default: {default_option}]: ").strip()

        if not choice:  # Use default based on platform
            if is_windows:
                print(f"   ✓ Using 1 core (sequential - recommended for Windows)")
            else:
                print(f"   ✓ Using all {cpu_count} cores")
            return default_cores
        elif choice == '1':
            print(f"   ✓ Using 1 core (sequential)")
            return 1
        elif choice == '2':
            cores = max(1, cpu_count // 2)
            if is_windows:
                print(f"   ⚠️  Attempting {cores} cores (may fallback to 1 if multiprocessing fails)")
            else:
                print(f"   ✓ Using {cores} cores")
            return cores
        elif choice == '3':
            if is_windows:
                print(f"   ⚠️  Attempting all {cpu_count} cores (may fallback to 1 if multiprocessing fails)")
            else:
                print(f"   ✓ Using all {cpu_count} cores")
            return -1
        elif choice == '4':
            while True:
                custom = input(f"   Enter number of cores (1-{cpu_count}): ").strip()
                try:
                    n_jobs = int(custom)
                    if 1 <= n_jobs <= cpu_count:
                        if is_windows and n_jobs > 1:
                            print(f"   ⚠️  Attempting {n_jobs} cores (may fallback to 1 if multiprocessing fails)")
                        else:
                            print(f"   ✓ Using {n_jobs} cores")
                        return n_jobs
                    else:
                        print(f"   ❌ Invalid. Enter 1-{cpu_count}")
                except ValueError:
                    print("   ❌ Invalid input. Enter a number.")
        else:
            if is_windows:
                print(f"   Invalid choice. Using default (1 core - recommended for Windows)")
                return 1
            else:
                print(f"   Invalid choice. Using default (all {cpu_count} cores)")
                return -1

    def run_all_approach_a(self):
        """Run all Approach A models"""
        self.clear_screen()
        self.print_header("RUN ALL APPROACH A MODELS")

        print("This will:")
        print("  • Extract features from 4 DINOv2 models (~1-2 hours)")
        print("  • Train 12 classifiers (~5-9 hours)")
        print("  • Total estimated time: 6-10 hours")
        print()
        print("Models already completed will be skipped.")
        print()

        confirm = input("Continue? (y/N): ").strip().lower()

        if confirm == 'y':
            # Configure parallel processing
            print("\nSome classifiers (SVM, Random Forest) support parallel processing.")
            n_jobs = self._get_n_jobs_from_user()

            print("\n🚀 Starting Approach A full pipeline...\n")
            results = self.orchestrator.train_approach_a_full(n_jobs=n_jobs)

            print(f"\n{'='*70}")
            print(f"RESULTS:")
            print(f"  Successful: {results['successful']}")
            print(f"  Failed: {results['failed']}")
            print(f"{'='*70}")

            input("\nPress Enter to continue...")

        self.approach_a_menu()

    def custom_approach_a_batch(self):
        """Custom batch training for Approach A"""
        self.clear_screen()
        self.print_header("CUSTOM APPROACH A BATCH")

        # Select extractors
        extractors = self.orchestrator.FEATURE_EXTRACTORS
        print("Select Feature Extractors (comma-separated, e.g., 1,3):")
        for i, extractor in enumerate(extractors, 1):
            print(f"  {i}. {extractor}")
        print()

        extractor_input = input("Extractors: ").strip()
        if not extractor_input:
            self.approach_a_menu()
            return

        try:
            extractor_indices = [int(x.strip()) - 1 for x in extractor_input.split(',')]
            selected_extractors = [extractors[i] for i in extractor_indices if 0 <= i < len(extractors)]

            if not selected_extractors:
                input("\n❌ No valid extractors selected. Press Enter to continue...")
                self.approach_a_menu()
                return

            # Select classifiers
            classifiers = self.orchestrator.CLASSIFIERS
            print(f"\nSelect Classifiers (comma-separated, e.g., 1,2 or 'all'):")
            for i, classifier in enumerate(classifiers, 1):
                print(f"  {i}. {classifier}")
            print()

            classifier_input = input("Classifiers: ").strip().lower()
            if not classifier_input:
                self.approach_a_menu()
                return

            if classifier_input == 'all':
                selected_classifiers = classifiers
            else:
                classifier_indices = [int(x.strip()) - 1 for x in classifier_input.split(',')]
                selected_classifiers = [classifiers[i] for i in classifier_indices if 0 <= i < len(classifiers)]

            if not selected_classifiers:
                input("\n❌ No valid classifiers selected. Press Enter to continue...")
                self.approach_a_menu()
                return

            # Confirm
            total_models = len(selected_extractors) * len(selected_classifiers)
            print(f"\nWill train {total_models} models:")
            print(f"  Extractors: {', '.join(selected_extractors)}")
            print(f"  Classifiers: {', '.join(selected_classifiers)}")
            print()

            # Check if any models are already completed
            completed_models = []
            for extractor in selected_extractors:
                for classifier in selected_classifiers:
                    model_id = f"{extractor}_{classifier}"
                    status = self.orchestrator.state['approach_a']['models'][model_id]['status']
                    if status == 'completed':
                        completed_models.append(model_id)

            # Ask about retraining if any are completed
            force = False
            if completed_models:
                print(f"\n⚠️  Found {len(completed_models)} already trained model(s):")
                # Show first 3 models if many, otherwise show all
                if len(completed_models) <= 3:
                    print(f"    {', '.join(completed_models)}")
                else:
                    print(f"    {', '.join(completed_models[:3])}, ... and {len(completed_models)-3} more")
                retrain_all = input("Retrain already completed models? (y/N): ").strip().lower()
                force = (retrain_all == 'y')
                print()

            confirm = input("Continue? (y/N): ").strip().lower()

            if confirm == 'y':
                # Configure parallel processing
                print("\nSome classifiers (SVM, Random Forest) support parallel processing.")
                n_jobs = self._get_n_jobs_from_user()

                print("\n🚀 Starting custom batch training...\n")
                results = self.orchestrator.train_approach_a_full(
                    extractors=selected_extractors,
                    classifiers=selected_classifiers,
                    n_jobs=n_jobs,
                    force=force
                )

                print(f"\n{'='*70}")
                print(f"RESULTS:")
                print(f"  Successful: {results['successful']}")
                print(f"  Failed: {results['failed']}")
                print(f"{'='*70}")

                input("\nPress Enter to continue...")

        except (ValueError, IndexError) as e:
            input(f"\n❌ Invalid input: {e}. Press Enter to continue...")

        self.approach_a_menu()

    def approach_b_menu(self):
        """Approach B submenu"""
        self.clear_screen()
        self.print_header("APPROACH B - Fine-Tuning")

        print("OPTIONS:")
        print("  1. Fine-tune Single Model")
        print("  2. Fine-tune Multiple Models")
        print("  3. Run All Approach B (4 Models)")
        print("  0. Back to Main Menu")
        print()

        choice = input("Select option (0-3): ").strip()

        if choice == '1':
            self.finetune_single_model()
        elif choice == '2':
            self.finetune_multiple_models()
        elif choice == '3':
            self.run_all_approach_b()
        elif choice == '0':
            return
        else:
            input("\n❌ Invalid option. Press Enter to continue...")
            self.approach_b_menu()

    def finetune_single_model(self):
        """Fine-tune a single Approach B model"""
        self.clear_screen()
        self.print_header("FINE-TUNE SINGLE MODEL")

        models = self.orchestrator.FINETUNE_MODELS

        print("Select Model:")
        for i, model in enumerate(models, 1):
            status = self.orchestrator.state['approach_b']['models'][model]['status']
            icon = '✅' if status == 'completed' else '⏳' if status == 'pending' else '🔄' if status == 'in_progress' else '❌'
            print(f"  {i}. {icon} {model}")
        print("  0. Back")
        print()

        choice = input("Select model (0-4): ").strip()

        if choice == '0':
            self.approach_b_menu()
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model = models[idx]
                status = self.orchestrator.state['approach_b']['models'][model]['status']

                # GPU memory info
                gpu_memory = {
                    'plant_pretrained_base': '~10 GB',
                    'imagenet_small': '~6 GB',
                    'imagenet_base': '~10 GB',
                    'imagenet_large': '~16 GB'
                }

                print(f"\n📋 Model: {model}")
                print(f"⏱️  Estimated time: 2-6 hours")
                print(f"💾 GPU memory needed: {gpu_memory.get(model, '~10 GB')}")
                print()

                force = False
                if status == 'completed':
                    rerun = input("Model already trained. Retrain? (y/N): ").strip().lower()
                    force = (rerun == 'y')
                    if not force:
                        input("\nSkipping. Press Enter to continue...")
                        self.approach_b_menu()
                        return

                epochs_input = input("Number of epochs (default=60): ").strip()
                epochs = int(epochs_input) if epochs_input else 60

                print(f"\n🔄 Starting fine-tuning for {model}...")
                print("Training output will be shown in real-time.\n")

                success = self.orchestrator.finetune_model(model, epochs=epochs, force=force)

                if success:
                    input("\n✅ Fine-tuning complete! Press Enter to continue...")
                else:
                    input("\n❌ Fine-tuning failed. Press Enter to continue...")

                self.approach_b_menu()
            else:
                input("\n❌ Invalid selection. Press Enter to continue...")
                self.finetune_single_model()

        except ValueError:
            input("\n❌ Invalid input. Press Enter to continue...")
            self.finetune_single_model()

    def finetune_multiple_models(self):
        """Fine-tune multiple Approach B models"""
        self.clear_screen()
        self.print_header("FINE-TUNE MULTIPLE MODELS")

        models = self.orchestrator.FINETUNE_MODELS

        print("Select Models (comma-separated, e.g., 1,3):")
        for i, model in enumerate(models, 1):
            status = self.orchestrator.state['approach_b']['models'][model]['status']
            icon = '✅' if status == 'completed' else '⏳'
            print(f"  {i}. {icon} {model}")
        print()

        model_input = input("Models: ").strip()
        if not model_input:
            self.approach_b_menu()
            return

        try:
            model_indices = [int(x.strip()) - 1 for x in model_input.split(',')]
            selected_models = [models[i] for i in model_indices if 0 <= i < len(models)]

            if not selected_models:
                input("\n❌ No valid models selected. Press Enter to continue...")
                self.approach_b_menu()
                return

            print(f"\nWill fine-tune {len(selected_models)} models:")
            for model in selected_models:
                print(f"  • {model}")
            print(f"\nEstimated time: {len(selected_models) * 2}-{len(selected_models) * 6} hours")
            print()

            # Check if any selected models are already completed
            completed_models = []
            for model in selected_models:
                status = self.orchestrator.state['approach_b']['models'][model]['status']
                if status == 'completed':
                    completed_models.append(model)

            # Ask about retraining if any are completed
            force = False
            if completed_models:
                print(f"\n⚠️  Found {len(completed_models)} already trained model(s): {', '.join(completed_models)}")
                retrain = input("Retrain already completed models? (y/N): ").strip().lower()
                force = (retrain == 'y')

            epochs_input = input("Number of epochs per model (default=60): ").strip()
            epochs = int(epochs_input) if epochs_input else 60

            confirm = input("Continue? (y/N): ").strip().lower()

            if confirm == 'y':
                print("\n🚀 Starting fine-tuning pipeline...\n")
                results = self.orchestrator.train_approach_b_full(
                    models=selected_models,
                    epochs=epochs,
                    force=force
                )

                print(f"\n{'='*70}")
                print(f"RESULTS:")
                print(f"  Successful: {results['successful']}")
                print(f"  Failed: {results['failed']}")
                print(f"{'='*70}")

                input("\nPress Enter to continue...")

        except (ValueError, IndexError) as e:
            input(f"\n❌ Invalid input: {e}. Press Enter to continue...")

        self.approach_b_menu()

    def run_all_approach_b(self):
        """Run all Approach B models"""
        self.clear_screen()
        self.print_header("RUN ALL APPROACH B MODELS")

        print("This will fine-tune all 4 DINOv2 models:")
        models = self.orchestrator.FINETUNE_MODELS
        for model in models:
            status = self.orchestrator.state['approach_b']['models'][model]['status']
            icon = '✅' if status == 'completed' else '⏳'
            print(f"  {icon} {model}")
        print()
        print("Estimated time: 8-24 hours")
        print()

        # Check if any models are already completed
        completed_models = [m for m in models if self.orchestrator.state['approach_b']['models'][m]['status'] == 'completed']

        force = False
        if completed_models:
            print(f"⚠️  Found {len(completed_models)} already trained model(s): {', '.join(completed_models)}")
            retrain = input("Retrain already completed models? (y/N): ").strip().lower()
            force = (retrain == 'y')
            print()

        epochs_input = input("Number of epochs per model (default=60): ").strip()
        epochs = int(epochs_input) if epochs_input else 60

        confirm = input("\nContinue? (y/N): ").strip().lower()

        if confirm == 'y':
            print("\n🚀 Starting Approach B full pipeline...\n")
            results = self.orchestrator.train_approach_b_full(epochs=epochs, force=force)

            print(f"\n{'='*70}")
            print(f"RESULTS:")
            print(f"  Successful: {results['successful']}")
            print(f"  Failed: {results['failed']}")
            print(f"{'='*70}")

            input("\nPress Enter to continue...")

        self.approach_b_menu()

    def run_full_pipeline(self):
        """Run complete pipeline (all 16 models)"""
        self.clear_screen()
        self.print_header("RUN FULL PIPELINE - ALL 16 MODELS")

        print("This will train:")
        print("  • Approach A: 12 models (feature extraction + classifiers)")
        print("  • Approach B: 4 models (fine-tuning)")
        print()
        print("Estimated time: 14-34 hours")
        print("Models already completed will be skipped.")
        print()

        confirm = input("Continue? (y/N): ").strip().lower()

        if confirm == 'y':
            # Configure parallel processing for Approach A
            print("\nSome classifiers (SVM, Random Forest) support parallel processing.")
            n_jobs = self._get_n_jobs_from_user()

            print("\n🚀 Starting FULL PIPELINE...\n")

            # Run Approach A
            print("\n" + "="*70)
            print("PHASE 1: APPROACH A")
            print("="*70 + "\n")
            a_results = self.orchestrator.train_approach_a_full(n_jobs=n_jobs)

            # Run Approach B
            print("\n" + "="*70)
            print("PHASE 2: APPROACH B")
            print("="*70 + "\n")
            b_results = self.orchestrator.train_approach_b_full()

            # Final summary
            print("\n" + "="*70)
            print("🎉 FULL PIPELINE COMPLETE!")
            print("="*70)
            print(f"Approach A: {a_results['successful']}/12 successful, {a_results['failed']} failed")
            print(f"Approach B: {b_results['successful']}/4 successful, {b_results['failed']} failed")
            print(f"Total: {a_results['successful'] + b_results['successful']}/16 models trained")
            print("="*70)

            input("\nPress Enter to continue...")

    def view_detailed_status(self):
        """View detailed training status"""
        self.clear_screen()
        self.print_header("DETAILED TRAINING STATUS")

        state = self.orchestrator.state

        # Approach A Features
        print("APPROACH A - FEATURE EXTRACTION:")
        print("-" * 70)
        for extractor, data in state['approach_a']['features'].items():
            status = data['status']
            icon = '✅' if status == 'completed' else '🔄' if status == 'in_progress' else '❌' if status == 'failed' else '⏳'
            print(f"{icon} {extractor:20s} - {status}")
            if data.get('error'):
                print(f"   Error: {data['error'][:60]}...")
        print()

        # Approach A Models
        print("APPROACH A - CLASSIFIERS:")
        print("-" * 70)
        for model_id, data in state['approach_a']['models'].items():
            status = data['status']
            icon = '✅' if status == 'completed' else '🔄' if status == 'in_progress' else '❌' if status == 'failed' else '⏳'
            print(f"{icon} {model_id:35s} - {status}")
            if data.get('error'):
                print(f"   Error: {data['error'][:60]}...")
        print()

        # Approach B Models
        print("APPROACH B - FINE-TUNED:")
        print("-" * 70)
        for model, data in state['approach_b']['models'].items():
            status = data['status']
            icon = '✅' if status == 'completed' else '🔄' if status == 'in_progress' else '❌' if status == 'failed' else '⏳'
            print(f"{icon} {model:30s} - {status}")
            if data.get('error'):
                print(f"   Error: {data['error'][:60]}...")
        print()

        input("Press Enter to continue...")

    def generate_reports(self):
        """Generate detailed evaluation reports with domain breakdown"""
        self.clear_screen()
        self.print_header("GENERATE DETAILED EVALUATION REPORTS")

        print("Available Reports:")
        print("  1. Evaluate Approach A (Feature Extraction + Classifiers)")
        print("  2. Evaluate Approach B (Fine-Tuned Models)")
        print("  3. Evaluate All Models (Approach A + B)")
        print("  4. View Saved Results (Approach A)")
        print("  5. View Saved Results (Approach B)")
        print("  0. Back")
        print()

        choice = input("Select option (0-5): ").strip()

        if choice == '1':
            self.evaluate_approach_a_detailed()
        elif choice == '2':
            self.evaluate_approach_b_detailed()
        elif choice == '3':
            self.evaluate_all_models_detailed()
        elif choice == '4':
            self.view_saved_results('a')
        elif choice == '5':
            self.view_saved_results('b')
        elif choice == '0':
            return
        else:
            input("\n❌ Invalid option. Press Enter to continue...")
            self.generate_reports()

    def evaluate_approach_a_detailed(self):
        """Run detailed evaluation for Approach A models"""
        import subprocess
        self.clear_screen()
        self.print_header("EVALUATE APPROACH A - DETAILED")

        # Check if models exist
        results_dir = Path("Approach_A_Feature_Extraction/results")
        if not results_dir.exists() or not list(results_dir.iterdir()):
            print("❌ No trained models found in Approach A results directory!")
            input("\nPress Enter to continue...")
            self.generate_reports()
            return

        print("This will evaluate ALL trained Approach A models with:")
        print("  • Top-1 Accuracy")
        print("  • Top-5 Accuracy")
        print("  • Mean Reciprocal Rank (MRR)")
        print()
        print("Breakdown by:")
        print("  • Overall (all 207 test samples)")
        print("  • With Pairs (60 classes with both herbarium + field)")
        print("  • Without Pairs (40 classes with herbarium only)")
        print()

        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            self.generate_reports()
            return

        print("\n🔄 Running detailed evaluation...")
        print("This may take a few minutes.\n")

        # Run evaluation script
        script_path = Path("Approach_A_Feature_Extraction/evaluate_classifiers_detailed.py")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path(__file__).parent)
        )

        if result.returncode == 0:
            print("\n✅ Evaluation complete!")
            print(f"📁 Results saved in: Approach_A_Feature_Extraction/evaluation_results/")

            # Offer to view detailed results
            view = input("\nView detailed results now? (y/N): ").strip().lower()
            if view == 'y':
                self.view_saved_results('a')
            else:
                self.generate_reports()
        else:
            print("\n❌ Evaluation failed!")
            input("\nPress Enter to continue...")
            self.generate_reports()

    def evaluate_approach_b_detailed(self):
        """Run detailed evaluation for Approach B models"""
        import subprocess
        self.clear_screen()
        self.print_header("EVALUATE APPROACH B - DETAILED")

        # Check if models exist
        models_dir = Path("Approach_B_Fine_Tuning/Models")
        if not models_dir.exists() or not list(models_dir.iterdir()):
            print("❌ No trained models found in Approach B Models directory!")
            input("\nPress Enter to continue...")
            self.generate_reports()
            return

        print("This will evaluate ALL trained Approach B models with:")
        print("  • Top-1 Accuracy")
        print("  • Top-5 Accuracy")
        print("  • Mean Reciprocal Rank (MRR)")
        print()
        print("Breakdown by:")
        print("  • Overall (all 207 test samples)")
        print("  • With Pairs (60 classes with both herbarium + field)")
        print("  • Without Pairs (40 classes with herbarium only)")
        print()

        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            self.generate_reports()
            return

        print("\n🔄 Running detailed evaluation...")
        print("This may take a few minutes.\n")

        # Run evaluation script
        script_path = Path("Approach_B_Fine_Tuning/evaluate_all_models_detailed.py")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path(__file__).parent)
        )

        if result.returncode == 0:
            print("\n✅ Evaluation complete!")
            print(f"📁 Results saved in: Approach_B_Fine_Tuning/evaluation_results/")

            # Offer to view detailed results
            view = input("\nView detailed results now? (y/N): ").strip().lower()
            if view == 'y':
                self.view_saved_results('b')
            else:
                self.generate_reports()
        else:
            print("\n❌ Evaluation failed!")
            input("\nPress Enter to continue...")
            self.generate_reports()

    def evaluate_all_models_detailed(self):
        """Run detailed evaluation for ALL models (A + B)"""
        import subprocess
        self.clear_screen()
        self.print_header("EVALUATE ALL MODELS - DETAILED")

        print("This will evaluate:")
        print("  • Approach A: All trained classifiers")
        print("  • Approach B: All fine-tuned models")
        print()
        print("Estimated time: 5-10 minutes")
        print()

        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            self.generate_reports()
            return

        # Run Approach A evaluation
        print("\n" + "=" * 70)
        print("EVALUATING APPROACH A")
        print("=" * 70)
        script_path_a = Path("Approach_A_Feature_Extraction/evaluate_classifiers_detailed.py")
        subprocess.run([sys.executable, str(script_path_a)], cwd=str(Path(__file__).parent))

        # Run Approach B evaluation
        print("\n" + "=" * 70)
        print("EVALUATING APPROACH B")
        print("=" * 70)
        script_path_b = Path("Approach_B_Fine_Tuning/evaluate_all_models_detailed.py")
        subprocess.run([sys.executable, str(script_path_b)], cwd=str(Path(__file__).parent))

        print("\n✅ All evaluations complete!")
        input("\nPress Enter to continue...")
        self.generate_reports()

    def view_saved_results(self, approach: str):
        """Display saved evaluation results with model selection"""
        self.clear_screen()

        if approach == 'a':
            title = "APPROACH A - EVALUATION RESULTS"
            results_file = "Approach_A_Feature_Extraction/evaluation_results/detailed_results.json"
        else:
            title = "APPROACH B - EVALUATION RESULTS"
            results_file = "Approach_B_Fine_Tuning/evaluation_results/detailed_results.json"

        self.print_header(title)

        if not Path(results_file).exists():
            print("❌ No evaluation results found!")
            print(f"Expected file: {results_file}")
            print("\nRun evaluation first from the 'Generate Reports' menu.")
            input("\nPress Enter to continue...")
            self.generate_reports()
            return

        # Load results
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            input("\nPress Enter to continue...")
            self.generate_reports()
            return

        if not results:
            print("❌ No models evaluated yet!")
            input("\nPress Enter to continue...")
            self.generate_reports()
            return

        # Display summary table
        print("SUMMARY TABLE:")
        print("-" * 90)
        self.display_results_summary(results)
        print("-" * 90)
        print()

        # Offer detailed view
        print("OPTIONS:")
        print("  1. View detailed breakdown for specific model")
        print("  0. Back")
        print()

        choice = input("Select option (0-1): ").strip()

        if choice == '1':
            self.view_model_detailed(results, approach)
        else:
            self.generate_reports()

    def display_results_summary(self, results: dict):
        """Display formatted summary table for all models"""
        try:
            from tabulate import tabulate
        except ImportError:
            print("⚠️  tabulate package not installed. Showing basic table.")
            print("\nInstall with: pip install tabulate\n")
            # Fallback to basic formatting
            print(f"{'Model':<40} {'N':>5} {'Top-1 Acc':>12} {'Top-5 Acc':>12} {'MRR':>15}")
            print("-" * 90)
            for model_name, metrics in results.items():
                overall = metrics.get('overall', {})
                print(f"{model_name:<40} {overall.get('N', 0):>5} "
                      f"{overall.get('top1', 0):>11.2f}% {overall.get('top5', 0):>11.2f}% "
                      f"{overall.get('mrr', 0):>15.10f}")
            return

        # Prepare table data
        table_data = []
        for model_name, metrics in results.items():
            # Overall metrics
            overall = metrics.get('overall', {})
            table_data.append([
                model_name,
                overall.get('N', 0),
                f"{overall.get('top1', 0):.2f}%",
                f"{overall.get('top5', 0):.2f}%",
                f"{overall.get('mrr', 0):.10f}"
            ])

        headers = ['Model', 'N', 'Top-1 Acc', 'Top-5 Acc', 'MRR']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))

    def view_model_detailed(self, results: dict, approach: str):
        """View detailed metrics for a specific model"""
        self.clear_screen()
        self.print_header(f"{'APPROACH A' if approach == 'a' else 'APPROACH B'} - MODEL DETAILS")

        # List available models
        models = list(results.keys())
        print("Select Model:\n")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        print("\n  0. Back")
        print()

        try:
            choice = input(f"Select model (0-{len(models)}): ").strip()
            if choice == '0':
                self.view_saved_results(approach)
                return

            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_name = models[idx]
                metrics = results[model_name]

                # Display detailed breakdown
                print(f"\n{'=' * 90}")
                print(f"MODEL: {model_name}")
                print('=' * 90)
                print()

                # Create detailed table
                try:
                    from tabulate import tabulate
                    table_data = [
                        ['Overall', metrics['overall']['N'],
                         f"{metrics['overall']['top1']:.2f}%",
                         f"{metrics['overall']['top5']:.2f}%",
                         f"{metrics['overall']['mrr']:.10f}"],
                        ['With Pairs', metrics['with_pairs']['N'],
                         f"{metrics['with_pairs']['top1']:.2f}%",
                         f"{metrics['with_pairs']['top5']:.2f}%",
                         f"{metrics['with_pairs']['mrr']:.10f}"],
                        ['Without Pairs', metrics['without_pairs']['N'],
                         f"{metrics['without_pairs']['top1']:.2f}%",
                         f"{metrics['without_pairs']['top5']:.2f}%",
                         f"{metrics['without_pairs']['mrr']:.10f}"]
                    ]

                    headers = ['Category', 'N', 'Top-1 Acc', 'Top-5 Acc', 'MRR']
                    print(tabulate(table_data, headers=headers, tablefmt='grid'))
                except ImportError:
                    # Fallback formatting
                    print(f"{'Category':<15} {'N':>5} {'Top-1 Acc':>12} {'Top-5 Acc':>12} {'MRR':>15}")
                    print("-" * 70)
                    for cat in ['overall', 'with_pairs', 'without_pairs']:
                        m = metrics[cat]
                        cat_name = cat.replace('_', ' ').title()
                        print(f"{cat_name:<15} {m['N']:>5} {m['top1']:>11.2f}% "
                              f"{m['top5']:>11.2f}% {m['mrr']:>15.10f}")
                print()

                # Analysis
                diff = metrics['with_pairs']['top1'] - metrics['without_pairs']['top1']
                print("CROSS-DOMAIN ANALYSIS:")
                if abs(diff) < 1.0:
                    print(f"  ✓ Excellent generalization (only {abs(diff):.2f}% difference)")
                elif abs(diff) < 5.0:
                    print(f"  → Good generalization ({abs(diff):.2f}% difference)")
                else:
                    print(f"  ⚠ Notable performance gap ({abs(diff):.2f}% difference)")
                print()

            input("\nPress Enter to continue...")
            self.view_model_detailed(results, approach)

        except ValueError:
            print("\n❌ Invalid input")
            input("Press Enter to continue...")
            self.view_model_detailed(results, approach)

    def reset_status_menu(self):
        """Reset model status menu"""
        self.clear_screen()
        self.print_header("RESET MODEL STATUS")

        print("⚠️  WARNING: This will reset model training status.")
        print("Files will not be deleted, only status tracking will be reset.")
        print()
        print("  1. Reset All Status")
        print("  2. Reset Specific Model")
        print("  0. Back")
        print()

        choice = input("Select option (0-2): ").strip()

        if choice == '1':
            confirm = input("\n⚠️  Reset ALL model statuses? (y/N): ").strip().lower()
            if confirm == 'y':
                self.orchestrator.reset_all_status()
                input("\n✅ All statuses reset. Press Enter to continue...")
        elif choice == '2':
            print("\nThis feature will be implemented if needed.")
            input("Press Enter to continue...")
        elif choice == '0':
            return

    def visualization_menu(self):
        """Visualization generation menu"""
        self.clear_screen()
        self.print_header("GENERATE VISUALIZATIONS")

        print("VISUALIZATION OPTIONS:")
        print("  1. Generate for single model")
        print("  2. Generate for all trained models")
        print("  3. Generate for specific extractor (all classifiers)")
        print("  0. Back to Main Menu")
        print()

        choice = input("Select option (0-3): ").strip()

        if choice == '1':
            self.generate_visualization_single()
        elif choice == '2':
            self.generate_visualization_all()
        elif choice == '3':
            self.generate_visualization_by_extractor()
        elif choice == '0':
            return
        else:
            input("\n❌ Invalid option. Press Enter to continue...")
            self.visualization_menu()

    def generate_visualization_single(self):
        """Generate visualizations for a single model"""
        self.clear_screen()
        self.print_header("GENERATE VISUALIZATIONS - SINGLE MODEL")

        # Get list of trained models
        results_dir = Path("Approach_A_Feature_Extraction/results")
        if not results_dir.exists():
            print("❌ No results directory found. Train some models first!")
            input("\nPress Enter to continue...")
            return

        model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            print("❌ No trained models found in results directory!")
            input("\nPress Enter to continue...")
            return

        # Display available models
        print("Available trained models:\n")
        for i, model_dir in enumerate(model_dirs, 1):
            # Check if visualizations exist
            viz_dir = model_dir / "visualizations"
            viz_exists = " [visualizations exist]" if viz_dir.exists() else ""
            print(f"  {i}. {model_dir.name}{viz_exists}")

        print("\n  0. Back")
        print()

        try:
            choice = input("Select model (0-{0}): ".format(len(model_dirs))).strip()
            if choice == '0':
                return

            idx = int(choice) - 1
            if 0 <= idx < len(model_dirs):
                model_dir = model_dirs[idx]

                # Infer classifier type and feature extractor
                from Approach_A_Feature_Extraction.generate_visualizations import infer_classifier_type, infer_feature_extractor

                classifier_type = infer_classifier_type(model_dir.name)
                feature_extractor = infer_feature_extractor(model_dir.name)

                if feature_extractor is None:
                    print(f"\n❌ Could not infer feature extractor from {model_dir.name}")
                    input("Press Enter to continue...")
                    return

                features_dir = Path(f"Approach_A_Feature_Extraction/features/{feature_extractor}")

                print(f"\nGenerating visualizations for: {model_dir.name}")
                print(f"Classifier type: {classifier_type}")
                print(f"Feature extractor: {feature_extractor}")
                print()

                try:
                    from Approach_A_Feature_Extraction.visualize_classifier import generate_all_visualizations
                    viz_paths = generate_all_visualizations(str(model_dir), str(features_dir), classifier_type)

                    print(f"\n✅ Generated {len(viz_paths)} visualizations!")
                    print(f"Saved in: {model_dir / 'visualizations'}")

                except Exception as e:
                    print(f"\n❌ Error generating visualizations: {e}")

                input("\nPress Enter to continue...")
            else:
                print("\n❌ Invalid selection")
                input("Press Enter to continue...")
        except ValueError:
            print("\n❌ Invalid input")
            input("Press Enter to continue...")

    def generate_visualization_all(self):
        """Generate visualizations for all trained models"""
        self.clear_screen()
        self.print_header("GENERATE VISUALIZATIONS - ALL MODELS")

        print("⚙️  This will generate visualizations for ALL trained models.")
        print("This may take several minutes depending on the number of models.\n")

        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            return

        print("\n📊 Generating visualizations...\n")

        try:
            from Approach_A_Feature_Extraction.generate_visualizations import generate_for_all_models
            generate_for_all_models("Approach_A_Feature_Extraction/results",
                                   "Approach_A_Feature_Extraction/features")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        input("\nPress Enter to continue...")

    def generate_visualization_by_extractor(self):
        """Generate visualizations for all classifiers of a specific extractor"""
        self.clear_screen()
        self.print_header("GENERATE VISUALIZATIONS - BY EXTRACTOR")

        extractors = self.orchestrator.FEATURE_EXTRACTORS

        print("Select Feature Extractor:\n")
        for i, extractor in enumerate(extractors, 1):
            print(f"  {i}. {extractor}")
        print("\n  0. Back")
        print()

        try:
            choice = input(f"Select extractor (0-{len(extractors)}): ").strip()
            if choice == '0':
                return

            idx = int(choice) - 1
            if 0 <= idx < len(extractors):
                extractor = extractors[idx]
                features_dir = Path(f"Approach_A_Feature_Extraction/features/{extractor}")

                if not features_dir.exists():
                    print(f"\n❌ Features not extracted for {extractor}")
                    input("Press Enter to continue...")
                    return

                print(f"\nGenerating visualizations for all {extractor} classifiers...")
                print()

                results_dir = Path("Approach_A_Feature_Extraction/results")
                successful = 0
                failed = 0

                for classifier in self.orchestrator.CLASSIFIERS:
                    model_dir = results_dir / f"{classifier}_{extractor}"

                    if not model_dir.exists():
                        print(f"   ⚠️  {classifier}_{extractor} - not trained, skipping")
                        continue

                    try:
                        from Approach_A_Feature_Extraction.visualize_classifier import generate_all_visualizations
                        viz_paths = generate_all_visualizations(str(model_dir), str(features_dir), classifier)
                        print(f"   ✅ {classifier}_{extractor} - {len(viz_paths)} visualizations")
                        successful += 1
                    except Exception as e:
                        print(f"   ❌ {classifier}_{extractor} - Error: {e}")
                        failed += 1

                print(f"\n✅ Complete! Successful: {successful}, Failed: {failed}")
                input("\nPress Enter to continue...")
            else:
                print("\n❌ Invalid selection")
                input("Press Enter to continue...")
        except ValueError:
            print("\n❌ Invalid input")
            input("Press Enter to continue...")

    def crossdomain_menu(self):
        """Cross-Domain Workflow Menu"""
        import subprocess

        self.clear_screen()
        self.print_header("CROSS-DOMAIN WORKFLOW")

        print("Cross-Domain Validation Split for Testing Herbarium→Field Generalization")
        print()
        print("OPTIONS:")
        print("  1. Create Cross-Domain Validation Split")
        print("  2. Extract Features (Cross-Domain)")
        print("  3. Train Classifiers (Cross-Domain)")
        print("  4. Evaluate & Compare (Old vs New)")
        print("  5. View Cross-Domain Statistics")
        print("  6. Evaluate Validation Set (Detailed)")
        print("  7. Evaluate Test Set (Detailed)")
        print("  0. Back to Main Menu")
        print()

        choice = input("Select option (0-7): ").strip()

        if choice == '1':
            # Create cross-domain validation split
            print("\n🔄 Creating Cross-Domain Validation Split...")
            print("This will create:")
            print("  - Dataset/balanced_train_crossdomain/")
            print("  - Dataset/validation_crossdomain/")
            print()
            confirm = input("Proceed? (y/N): ").strip().lower()

            if confirm == 'y':
                result = subprocess.run(['python', 'Src/data_balancing_crossdomain.py'],
                                      capture_output=False, text=True)
                if result.returncode == 0:
                    print("\n✅ Cross-domain split created successfully!")
                else:
                    print("\n❌ Error creating cross-domain split")
                input("\nPress Enter to continue...")
            else:
                print("\n❌ Cancelled")
                input("Press Enter to continue...")
            self.crossdomain_menu()

        elif choice == '2':
            # Extract features with cross-domain split
            print("\n🔍 Extract Features (Cross-Domain Validation)")
            print()
            print("Select Model:")
            models = ['plant_pretrained_base', 'imagenet_small', 'imagenet_base', 'imagenet_large']
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            print("  5. All Models")
            print("  0. Back")
            print()

            model_choice = input("Select (0-5): ").strip()

            if model_choice == '0':
                self.crossdomain_menu()
                return
            elif model_choice == '5':
                selected_models = models
            elif model_choice in ['1', '2', '3', '4']:
                selected_models = [models[int(model_choice) - 1]]
            else:
                print("\n❌ Invalid choice")
                input("Press Enter to continue...")
                self.crossdomain_menu()
                return

            print(f"\n🔄 Extracting features for: {', '.join(selected_models)}")
            print("This will save to: Approach_A_Feature_Extraction/features_crossdomain/")
            print()

            for model in selected_models:
                print(f"\n📊 Extracting {model}...")
                batch_size = 48 if model == 'imagenet_large' else 64
                result = subprocess.run([
                    'python', 'Approach_A_Feature_Extraction/extract_features.py',
                    '--model_type', model,
                    '--train_dir', 'Dataset/balanced_train_crossdomain',
                    '--val_dir', 'Dataset/validation_crossdomain',
                    '--test_dir', 'Dataset/test',
                    '--output_dir', 'Approach_A_Feature_Extraction/features_crossdomain',
                    '--batch_size', str(batch_size)
                ], capture_output=False, text=True)

                if result.returncode == 0:
                    print(f"✅ {model} features extracted!")
                else:
                    print(f"❌ {model} extraction failed")

            input("\nPress Enter to continue...")
            self.crossdomain_menu()

        elif choice == '3':
            # Train classifiers with cross-domain features
            print("\n🏋️  Train Classifiers (Cross-Domain Features)")
            print()
            print("Select Feature Set:")
            models = ['plant_pretrained_base', 'imagenet_small', 'imagenet_base', 'imagenet_large']
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            print("  0. Back")
            print()

            model_choice = input("Select (0-4): ").strip()

            if model_choice == '0':
                self.crossdomain_menu()
                return
            elif model_choice in ['1', '2', '3', '4']:
                selected_model = models[int(model_choice) - 1]
            else:
                print("\n❌ Invalid choice")
                input("Press Enter to continue...")
                self.crossdomain_menu()
                return

            print(f"\nSelect Classifier:")
            classifiers = ['linear_probe', 'svm', 'logistic_regression']
            for i, clf in enumerate(classifiers, 1):
                print(f"  {i}. {clf}")
            print("  4. All Classifiers")
            print("  0. Back")
            print()

            clf_choice = input("Select (0-4): ").strip()

            if clf_choice == '0':
                self.crossdomain_menu()
                return
            elif clf_choice == '4':
                selected_classifiers = classifiers
            elif clf_choice in ['1', '2', '3']:
                selected_classifiers = [classifiers[int(clf_choice) - 1]]
            else:
                print("\n❌ Invalid choice")
                input("Press Enter to continue...")
                self.crossdomain_menu()
                return

            features_dir = f'Approach_A_Feature_Extraction/features_crossdomain/{selected_model}'

            if not os.path.exists(features_dir):
                print(f"\n❌ Features not found: {features_dir}")
                print("Please extract features first (Option 2)")
                input("\nPress Enter to continue...")
                self.crossdomain_menu()
                return

            for clf in selected_classifiers:
                print(f"\n🔄 Training {clf} on {selected_model}...")
                output_dir = f'Approach_A_Feature_Extraction/results_crossdomain/{clf}_{selected_model}'

                if clf == 'linear_probe':
                    script = 'Approach_A_Feature_Extraction/train_linear_probe.py'
                elif clf == 'svm':
                    script = 'Approach_A_Feature_Extraction/train_svm.py'
                elif clf == 'logistic_regression':
                    script = 'Approach_A_Feature_Extraction/train_logistic_regression.py'

                result = subprocess.run([
                    'python', script,
                    '--features_dir', features_dir,
                    '--output_dir', output_dir
                ], capture_output=False, text=True)

                if result.returncode == 0:
                    print(f"✅ {clf} trained successfully!")
                else:
                    print(f"❌ {clf} training failed")

            input("\nPress Enter to continue...")
            self.crossdomain_menu()

        elif choice == '4':
            # Evaluate and compare
            print("\n📊 Evaluate & Compare (Old vs New Validation)")
            print()

            # Load old validation metrics (random 80/20 split)
            old_results_dir = "Approach_A_Feature_Extraction/results"
            new_results_dir = "Approach_A_Feature_Extraction/results_crossdomain"
            test_eval_file = "Approach_A_Feature_Extraction/evaluation_results/detailed_results.json"

            print("="*80)
            print("VALIDATION METRICS COMPARISON")
            print("="*80)

            # Display old validation results
            print("\n[1] OLD VALIDATION (Random 80/20 Split)")
            print("-" * 80)
            if os.path.exists(old_results_dir):
                old_val_acc = {}
                for model_dir in os.listdir(old_results_dir):
                    metrics_file = os.path.join(old_results_dir, model_dir, "results/metrics_summary.json")
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            old_val_acc[model_dir] = metrics.get('best_val_accuracy', 'N/A')

                if old_val_acc:
                    print(f"{'Model':<40} {'Val Accuracy':>15}")
                    print("-" * 80)
                    for model, acc in sorted(old_val_acc.items()):
                        print(f"{model:<40} {acc:>14.2f}%" if isinstance(acc, (int, float)) else f"{model:<40} {acc:>15}")
                    print("\nNote: High accuracy (99.7-99.9%) is misleading - same distribution in train/val")
                else:
                    print("No old validation results found")
            else:
                print(f"Directory not found: {old_results_dir}")

            # Display new validation results (if available)
            print("\n[2] NEW VALIDATION (Cross-Domain Split)")
            print("-" * 80)
            if os.path.exists(new_results_dir):
                new_val_acc = {}
                for model_dir in os.listdir(new_results_dir):
                    metrics_file = os.path.join(new_results_dir, model_dir, "results/metrics_summary.json")
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            new_val_acc[model_dir] = metrics.get('best_val_accuracy', 'N/A')

                if new_val_acc:
                    print(f"{'Model':<40} {'Val Accuracy':>15}")
                    print("-" * 80)
                    for model, acc in sorted(new_val_acc.items()):
                        print(f"{model:<40} {acc:>14.2f}%" if isinstance(acc, (int, float)) else f"{model:<40} {acc:>15}")
                    print("\nNote: Lower accuracy expected - tests herb→field generalization")
                else:
                    print("No cross-domain validation results found yet")
                    print("Run Option 3 (Train Classifiers) first")
            else:
                print(f"Directory not found: {new_results_dir}")
                print("Cross-domain classifiers not trained yet")

            # Display test results (detailed breakdown)
            print("\n[3] TEST SET RESULTS (Detailed Breakdown)")
            print("-" * 80)
            if os.path.exists(test_eval_file):
                with open(test_eval_file, 'r') as f:
                    test_results = json.load(f)

                print(f"\n{'Model':<30} {'Category':<15} {'Top-1':>8} {'Top-5':>8} {'MRR':>8}")
                print("=" * 80)

                for model_name, categories in sorted(test_results.items()):
                    for i, (cat_name, metrics) in enumerate([
                        ('overall', categories.get('overall', {})),
                        ('with_pairs', categories.get('with_pairs', {})),
                        ('without_pairs', categories.get('without_pairs', {}))
                    ]):
                        model_col = model_name if i == 0 else ""
                        top1 = metrics.get('top1', 0)
                        top5 = metrics.get('top5', 0)
                        mrr = metrics.get('mrr', 0)
                        print(f"{model_col:<30} {cat_name:<15} {top1:>7.2f}% {top5:>7.2f}% {mrr:>8.3f}")
                    print("-" * 80)

                print("\nKey Insight:")
                print("  - Without-pairs shows 14-22% accuracy (catastrophic!)")
                print("  - Models fail on classes with only herbarium training data")
                print("  - Cross-domain split helps identify this issue during validation")
            else:
                print(f"Test evaluation not found: {test_eval_file}")
                print("Run: python Approach_A_Feature_Extraction/evaluate_classifiers_detailed.py")

            print("\n" + "="*80)
            input("\nPress Enter to continue...")
            self.crossdomain_menu()

        elif choice == '5':
            # View statistics
            print("\n📈 Cross-Domain Statistics")
            print()

            # Check if cross-domain datasets exist
            train_cd = "Dataset/balanced_train_crossdomain"
            val_cd = "Dataset/validation_crossdomain"

            if os.path.exists(train_cd) and os.path.exists(val_cd):
                print("✅ Cross-Domain Datasets:")
                print(f"   Training: {train_cd}")
                print(f"   Validation: {val_cd}")
                print()

                # Load metadata if available
                try:
                    with open(f"{train_cd}/metadata.json", 'r') as f:
                        train_meta = json.load(f)
                    with open(f"{val_cd}/metadata.json", 'r') as f:
                        val_meta = json.load(f)

                    print("Training Set:")
                    print(f"   Total: {train_meta['total_images']} images")
                    print(f"   Herbarium: {train_meta['herbarium_images']}")
                    print(f"   Field: {train_meta['field_images']}")
                    print()
                    print("Validation Set:")
                    print(f"   Total: {val_meta['total_images']} images")
                    print(f"   Herbarium: {val_meta['herbarium_images']}")
                    print(f"   Field: {val_meta['field_images']}")
                    print()
                    print("Key Insight:")
                    field_pct = val_meta['field_images'] / val_meta['total_images'] * 100
                    print(f"   {field_pct:.1f}% field images in validation")
                    print(f"   Tests herb→field generalization!")
                except:
                    print("   (Metadata not available)")
            else:
                print("❌ Cross-Domain datasets not found")
                print("   Run Option 1 to create them")

            print()
            input("Press Enter to continue...")
            self.crossdomain_menu()

        elif choice == '6':
            # Evaluate validation set with detailed breakdown
            print("\n📊 Evaluate Validation Set (Detailed Breakdown)")
            print()

            results_dir = "Approach_A_Feature_Extraction/results_crossdomain"

            if not os.path.exists(results_dir):
                print(f"❌ Cross-domain results directory not found: {results_dir}")
                print("   Train classifiers first (Option 3)")
                input("\nPress Enter to continue...")
                self.crossdomain_menu()
                return

            # List available trained models
            available_models = []
            for model_dir in os.listdir(results_dir):
                model_path = os.path.join(results_dir, model_dir, "best_model.pth")
                joblib_path = os.path.join(results_dir, model_dir, "best_model.joblib")
                if os.path.exists(model_path) or os.path.exists(joblib_path):
                    available_models.append(model_dir)

            if not available_models:
                print("❌ No trained models found")
                print("   Train classifiers first (Option 3)")
                input("\nPress Enter to continue...")
                self.crossdomain_menu()
                return

            print("Available models:")
            for i, model in enumerate(sorted(available_models), 1):
                print(f"  {i}. {model}")
            print(f"  {len(available_models) + 1}. All Models")
            print("  0. Back")
            print()

            model_choice = input(f"Select (0-{len(available_models) + 1}): ").strip()

            if model_choice == '0':
                self.crossdomain_menu()
                return
            elif model_choice == str(len(available_models) + 1):
                selected_models = sorted(available_models)
            elif model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models):
                selected_models = [sorted(available_models)[int(model_choice) - 1]]
            else:
                print("\n❌ Invalid choice")
                input("Press Enter to continue...")
                self.crossdomain_menu()
                return

            print(f"\n🔍 Evaluating: {', '.join(selected_models)}")
            print("="*80)

            # Run evaluation for each model
            for model_name in selected_models:
                print(f"\n[{model_name}]")
                print("-" * 80)

                # Load validation predictions
                val_pred_file = os.path.join(results_dir, model_name, "val_predictions_proba.npy")

                # Extract feature model name (e.g., "linear_probe_imagenet_small" -> "imagenet_small")
                if "plant_pretrained_base" in model_name:
                    feature_model = "plant_pretrained_base"
                elif "imagenet_small" in model_name:
                    feature_model = "imagenet_small"
                elif "imagenet_base" in model_name:
                    feature_model = "imagenet_base"
                elif "imagenet_large" in model_name:
                    feature_model = "imagenet_large"
                else:
                    feature_model = model_name.split('_')[-1]  # Fallback

                val_labels_file = f"Approach_A_Feature_Extraction/features_crossdomain/{feature_model}/val_labels.npy"
                val_features_file = f"Approach_A_Feature_Extraction/features_crossdomain/{feature_model}/val_features.npy"

                # Check if this is a linear probe model (PyTorch)
                pytorch_model_file = os.path.join(results_dir, model_name, "best_model.pth")
                config_file = os.path.join(results_dir, model_name, "training_config.json")
                is_linear_probe = os.path.exists(pytorch_model_file) and os.path.exists(config_file)

                if not os.path.exists(val_labels_file):
                    print(f"  ❌ Labels not found: {val_labels_file}")
                    continue

                try:
                    val_labels = np.load(val_labels_file)

                    # For linear probe: load model and run inference
                    if is_linear_probe:
                        if not os.path.exists(val_features_file):
                            print(f"  ❌ Features not found: {val_features_file}")
                            continue

                        import torch
                        import torch.nn as nn

                        # Define LinearProbe class
                        class LinearProbe(nn.Module):
                            def __init__(self, input_dim, num_classes):
                                super(LinearProbe, self).__init__()
                                self.fc = nn.Linear(input_dim, num_classes)

                            def forward(self, x):
                                return self.fc(x)

                        # Load features and model
                        val_features = np.load(val_features_file)

                        with open(config_file, 'r') as f:
                            config = json.load(f)

                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = LinearProbe(config['input_dim'], config['num_classes'])
                        model.load_state_dict(torch.load(pytorch_model_file, map_location=device, weights_only=True))
                        model = model.to(device)
                        model.eval()

                        # Run inference
                        X_tensor = torch.from_numpy(val_features).float().to(device)
                        with torch.no_grad():
                            outputs = model(X_tensor)
                            probs = torch.softmax(outputs, dim=1)

                        val_pred_proba = probs.cpu().numpy()

                    # For sklearn models: load saved predictions
                    else:
                        if not os.path.exists(val_pred_file):
                            print(f"  ⚠️  Predictions not saved during training: {val_pred_file}")
                            continue

                        val_pred_proba = np.load(val_pred_file)

                    # Load class categories
                    with_pairs_file = "Dataset/list/class_with_pairs.txt"
                    without_pairs_file = "Dataset/list/class_without_pairs.txt"

                    # Get class ID to index mapping (PlantCLEF IDs -> 0-99)
                    train_dir = "Dataset/balanced_train_crossdomain"
                    class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
                    plantclef_id_to_idx = {int(class_id): idx for idx, class_id in enumerate(class_dirs)}

                    with_pairs_plantclef = []
                    without_pairs_plantclef = []

                    if os.path.exists(with_pairs_file):
                        with open(with_pairs_file, 'r') as f:
                            with_pairs_plantclef = [int(line.strip()) for line in f if line.strip()]

                    if os.path.exists(without_pairs_file):
                        with open(without_pairs_file, 'r') as f:
                            without_pairs_plantclef = [int(line.strip()) for line in f if line.strip()]

                    # Map PlantCLEF IDs to 0-99 indices
                    with_pairs = [plantclef_id_to_idx[pid] for pid in with_pairs_plantclef if pid in plantclef_id_to_idx]
                    without_pairs = [plantclef_id_to_idx[pid] for pid in without_pairs_plantclef if pid in plantclef_id_to_idx]

                    # Compute metrics for each category
                    def compute_metrics(predictions, labels, indices=None):
                        if indices is not None and len(indices) > 0:
                            pred_subset = predictions[indices]
                            label_subset = labels[indices]
                        else:
                            pred_subset = predictions
                            label_subset = labels

                        if len(pred_subset) == 0:
                            return {"N": 0, "top1": 0, "top5": 0, "mrr": 0}

                        # Top-1
                        top1_pred = np.argmax(pred_subset, axis=1)
                        top1_acc = np.mean(top1_pred == label_subset) * 100

                        # Top-5
                        top5_pred = np.argsort(pred_subset, axis=1)[:, -5:]
                        top5_acc = np.mean([label_subset[i] in top5_pred[i] for i in range(len(label_subset))]) * 100

                        # MRR
                        mrr_scores = []
                        for i in range(len(pred_subset)):
                            sorted_indices = np.argsort(pred_subset[i])[::-1]
                            true_label = label_subset[i]
                            if true_label in sorted_indices:
                                rank_position = np.where(sorted_indices == true_label)[0][0] + 1
                                mrr_scores.append(1.0 / rank_position)
                            else:
                                mrr_scores.append(0.0)
                        mrr = np.mean(mrr_scores)

                        return {
                            "N": len(label_subset),
                            "top1": top1_acc,
                            "top5": top5_acc,
                            "mrr": mrr
                        }

                    # Categorize samples
                    with_pairs_indices = [i for i, label in enumerate(val_labels) if label in with_pairs]
                    without_pairs_indices = [i for i, label in enumerate(val_labels) if label in without_pairs]

                    # Compute metrics
                    overall = compute_metrics(val_pred_proba, val_labels)
                    with_pairs_metrics = compute_metrics(val_pred_proba, val_labels, with_pairs_indices)
                    without_pairs_metrics = compute_metrics(val_pred_proba, val_labels, without_pairs_indices)

                    # Display results
                    print(f"  {'Category':<15} {'N':>6} {'Top-1':>8} {'Top-5':>8} {'MRR':>8}")
                    print("  " + "-" * 55)
                    print(f"  {'Overall':<15} {overall['N']:>6} {overall['top1']:>7.2f}% {overall['top5']:>7.2f}% {overall['mrr']:>8.3f}")
                    print(f"  {'With-Pairs':<15} {with_pairs_metrics['N']:>6} {with_pairs_metrics['top1']:>7.2f}% {with_pairs_metrics['top5']:>7.2f}% {with_pairs_metrics['mrr']:>8.3f}")
                    print(f"  {'Without-Pairs':<15} {without_pairs_metrics['N']:>6} {without_pairs_metrics['top1']:>7.2f}% {without_pairs_metrics['top5']:>7.2f}% {without_pairs_metrics['mrr']:>8.3f}")

                except Exception as e:
                    print(f"  ❌ Error evaluating {model_name}: {e}")
                    import traceback
                    traceback.print_exc()

            print("\n" + "="*80)
            print("\nKey Insights:")
            print("  - With-Pairs tests herb→field generalization (field images in validation)")
            print("  - Without-Pairs is herb→herb baseline (herbarium only)")
            print("  - Compare to test set results to check for overfitting")
            print()
            input("Press Enter to continue...")
            self.crossdomain_menu()

        elif choice == '7':
            # Evaluate test set with detailed breakdown
            print("\n📊 Evaluate Test Set (Detailed Breakdown)")
            print()

            results_dir = "Approach_A_Feature_Extraction/results_crossdomain"
            features_base = "Approach_A_Feature_Extraction/features_crossdomain"
            output_dir = "Approach_A_Feature_Extraction/evaluation_results_crossdomain"

            if not os.path.exists(results_dir):
                print(f"❌ Results directory not found: {results_dir}")
                print("   Train classifiers first (Option 3)")
                input("\nPress Enter to continue...")
                self.crossdomain_menu()
                return

            # Check if test features exist
            test_features_exist = False
            for model_dir in os.listdir(features_base):
                test_feat_file = os.path.join(features_base, model_dir, "test_features.npy")
                if os.path.exists(test_feat_file):
                    test_features_exist = True
                    break

            if not test_features_exist:
                print("❌ Test features not found")
                print("   Extract features with test set first (Option 2)")
                input("\nPress Enter to continue...")
                self.crossdomain_menu()
                return

            print("🔄 Running detailed test evaluation...")
            print("This will evaluate all trained models on the test set with:")
            print("  - Overall / With-Pairs / Without-Pairs breakdown")
            print("  - Top-1, Top-5, MRR metrics")
            print()

            # Run evaluation script
            result = subprocess.run([
                'python', 'Approach_A_Feature_Extraction/evaluate_classifiers_detailed.py',
                '--results_dir', results_dir,
                '--features_base', features_base,
                '--with_pairs', 'Dataset/list/class_with_pairs.txt',
                '--without_pairs', 'Dataset/list/class_without_pairs.txt',
                '--groundtruth', 'Dataset/list/groundtruth.txt',
                '--output_dir', output_dir
            ], capture_output=False, text=True)

            if result.returncode == 0:
                print("\n✅ Evaluation complete!")
                print(f"📁 Results saved to: {output_dir}")

                # Display results if available
                detailed_results_file = os.path.join(output_dir, "detailed_results.json")
                if os.path.exists(detailed_results_file):
                    print("\n" + "="*80)
                    print("TEST SET RESULTS (Cross-Domain Models)")
                    print("="*80)

                    with open(detailed_results_file, 'r') as f:
                        test_results = json.load(f)

                    print(f"\n{'Model':<35} {'Category':<15} {'Top-1':>8} {'Top-5':>8} {'MRR':>8}")
                    print("=" * 80)

                    for model_name in sorted(test_results.keys()):
                        categories = test_results[model_name]
                        for i, (cat_name, metrics) in enumerate([
                            ('overall', categories.get('overall', {})),
                            ('with_pairs', categories.get('with_pairs', {})),
                            ('without_pairs', categories.get('without_pairs', {}))
                        ]):
                            model_col = model_name if i == 0 else ""
                            top1 = metrics.get('top1', 0)
                            top5 = metrics.get('top5', 0)
                            mrr = metrics.get('mrr', 0)
                            print(f"{model_col:<35} {cat_name:<15} {top1:>7.2f}% {top5:>7.2f}% {mrr:>8.3f}")
                        print("-" * 80)

                    print("\nKey Insights:")
                    print("  - Compare Test vs Validation to check for overfitting")
                    print("  - Without-Pairs shows real cross-domain challenge")
                    print("  - Use Option 4 to compare Old vs New validation approaches")
            else:
                print("\n❌ Evaluation failed")
                print("   Check error messages above")

            print()
            input("Press Enter to continue...")
            self.crossdomain_menu()

        elif choice == '0':
            return
        else:
            print("\n❌ Invalid option")
            input("Press Enter to continue...")
            self.crossdomain_menu()

    def run(self):
        """Main run loop"""
        while self.running:
            self.main_menu()


def main():
    """Entry point"""
    import subprocess  # Import here to avoid issues
    import sys

    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "="*70)
    print("  🌱 Plant Identification - Training Manager")
    print("  Easy terminal interface for model training")
    print("="*70 + "\n")

    manager = TrainingManager()
    manager.run()


if __name__ == "__main__":
    main()
