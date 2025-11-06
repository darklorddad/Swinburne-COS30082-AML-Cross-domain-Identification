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
        print("  1. Approach A - Feature Extraction & Classifiers")
        print("  2. Approach B - Fine-Tuning")
        print("  3. Run Full Pipeline (All 16 Models)")
        print("  4. View Detailed Status")
        print("  5. Generate Comparison Report")
        print("  6. Reset Model Status")
        print("  0. Exit")
        print()

        choice = input("Select option (0-6): ").strip()

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
                    self.orchestrator.train_classifier(extractor, classifier, force=force)

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
                    success = self.orchestrator.train_classifier(extractor, classifier, force=force)

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
            print("\n🚀 Starting Approach A full pipeline...\n")
            results = self.orchestrator.train_approach_a_full()

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
                print("\n🚀 Starting custom batch training...\n")
                results = self.orchestrator.train_approach_a_full(
                    extractors=selected_extractors,
                    classifiers=selected_classifiers
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
            print("\n🚀 Starting FULL PIPELINE...\n")

            # Run Approach A
            print("\n" + "="*70)
            print("PHASE 1: APPROACH A")
            print("="*70 + "\n")
            a_results = self.orchestrator.train_approach_a_full()

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
        """Generate comparison reports"""
        self.clear_screen()
        self.print_header("GENERATE COMPARISON REPORTS")

        print("Available Reports:")
        print("  1. Approach A Evaluation (all classifiers)")
        print("  2. Approach B Evaluation (all fine-tuned models)")
        print("  3. Both Reports")
        print("  0. Back")
        print()

        choice = input("Select option (0-3): ").strip()

        if choice == '1':
            self.run_evaluation('a')
        elif choice == '2':
            self.run_evaluation('b')
        elif choice == '3':
            self.run_evaluation('a')
            self.run_evaluation('b')
        elif choice == '0':
            return

    def run_evaluation(self, approach: str):
        """Run evaluation script"""
        if approach == 'a':
            script = "Approach_A_Feature_Extraction/evaluate_classifiers.py"
            title = "Approach A"
        else:
            script = "Approach_B_Fine_Tuning/evaluate_all_models.py"
            title = "Approach B"

        print(f"\n🔄 Running {title} evaluation...")

        script_path = Path(__file__).parent / script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path(__file__).parent)
        )

        if result.returncode == 0:
            print(f"✅ {title} evaluation complete!")
        else:
            print(f"❌ {title} evaluation failed")

        input("\nPress Enter to continue...")

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
