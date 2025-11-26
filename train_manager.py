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
        print(f"✅ {a_models['completed']}/16  ", end="")
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
        print("  1. Approach A - Feature Extraction & Classifiers")
        print("  2. Approach B - Fine-Tuning")
        print("  3. Run Full Pipeline (All 16 Models)")
        print("  4. View Detailed Status")
        print("  5. Generate Comparison Report")
        print("  6. Generate Visualizations")
        print("  7. Reset Model Status")
        print("  0. Exit")
        print()

        choice = input("Select option (0-7): ").strip()

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
                for classifier in classifiers:
                    model_id = f"{extractor}_{classifier}"
                    status = self.orchestrator.state['approach_a']['models'][model_id]['status']
                    if status == 'completed':
                        print(f"⏭️  Skipping {classifier} (already completed)")
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

            confirm = input("Continue? (y/N): ").strip().lower()

            if confirm == 'y':
                # Configure parallel processing
                print("\nSome classifiers (SVM, Random Forest) support parallel processing.")
                n_jobs = self._get_n_jobs_from_user()

                print("\n🚀 Starting custom batch training...\n")
                results = self.orchestrator.train_approach_a_full(
                    extractors=selected_extractors,
                    classifiers=selected_classifiers,
                    n_jobs=n_jobs
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

            epochs_input = input("Number of epochs per model (default=60): ").strip()
            epochs = int(epochs_input) if epochs_input else 60

            confirm = input("Continue? (y/N): ").strip().lower()

            if confirm == 'y':
                print("\n🚀 Starting fine-tuning pipeline...\n")
                results = self.orchestrator.train_approach_b_full(
                    models=selected_models,
                    epochs=epochs
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
        print("  • plant_pretrained_base")
        print("  • imagenet_small")
        print("  • imagenet_base")
        print("  • imagenet_large")
        print()
        print("Estimated time: 8-24 hours")
        print("Models already completed will be skipped.")
        print()

        epochs_input = input("Number of epochs per model (default=60): ").strip()
        epochs = int(epochs_input) if epochs_input else 60

        confirm = input("\nContinue? (y/N): ").strip().lower()

        if confirm == 'y':
            print("\n🚀 Starting Approach B full pipeline...\n")
            results = self.orchestrator.train_approach_b_full(epochs=epochs)

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

    def run(self):
        """Main run loop"""
        while self.running:
            self.main_menu()


def main():
    """Entry point"""
    import subprocess  # Import here to avoid issues

    print("\n" + "="*70)
    print("  🌱 Plant Identification - Training Manager")
    print("  Easy terminal interface for model training")
    print("="*70 + "\n")

    manager = TrainingManager()
    manager.run()


if __name__ == "__main__":
    main()
