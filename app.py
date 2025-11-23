import os
import gradio as gr
from gradio_wrapper import (
    classify_plant, show_model_charts, get_model_choices, update_model_choices,
    launch_autotrain_ui, launch_tensorboard, generate_manifest,
    split_dataset, check_dataset_balance, check_dataset_splittability,
    clean_dataset_names, save_metrics, evaluate_test_set, save_evaluation_results
)
try:
    from custom_utils import custom_sort_dataset, rename_test_images_func, evaluate_dataset
    CUSTOM_UTILS_AVAILABLE = True
except ImportError:
    CUSTOM_UTILS_AVAILABLE = False
    def custom_sort_dataset(*args, **kwargs): raise gr.Error("custom_utils not available")
    def rename_test_images_func(*args, **kwargs): raise gr.Error("custom_utils not available")
    def evaluate_dataset(*args, **kwargs): raise gr.Error("custom_utils not available")

DEFAULT_MANIFEST_PATH = os.path.join('core', 'manifest.md').replace(os.sep, '/')

# #############################################################################
# GRADIO UI
# #############################################################################

with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}") as demo:

    gr.HTML(
        """
        <script>
            window.addEventListener('load', () => {
                setInterval(() => {
                    const btn = document.getElementById('model_refresh_button');
                    if (btn) {
                        btn.click();
                    }
                }, 5000);
            });
        </script>
        """,
        visible=False
    )
    refresh_button = gr.Button(elem_id="model_refresh_button", visible=False)

    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    inf_source = gr.Radio(
                        choices=["Local", "Hugging Face Hub", "Local .pth"],
                        value="Local",
                        label="Model source"
                    )
                    
                    inf_model_path = gr.Dropdown(
                        label="Select local model", 
                        choices=[], 
                        value=None, 
                        filterable=False,
                        visible=True
                    )
                    
                    inf_hf_id = gr.Textbox(
                        label="Hugging Face model ID", 
                        placeholder="e.g. microsoft/resnet-50", 
                        visible=False
                    )
                    
                    with gr.Column(visible=False) as inf_pth_group:
                        inf_pth_file = gr.File(label="Upload .pth file", file_types=[".pth"])
                        inf_pth_classes = gr.File(label="Upload class list (txt/json)", file_types=[".txt", ".json"])
                        inf_pth_arch = gr.Textbox(
                            label="Architecture name (timm)", 
                            placeholder="e.g. resnet50, vit_base_patch16_224"
                        )

                inf_input_image = gr.Image(type="pil", label="Upload a plant image")

            with gr.Column(scale=1):
                inf_output_label = gr.Label(num_top_classes=5, label="Predictions")
                inf_button = gr.Button("Classify", variant="primary")

        def update_inf_inputs(source):
            return (
                gr.update(visible=(source == "Local")),
                gr.update(visible=(source == "Hugging Face Hub")),
                gr.update(visible=(source == "Local .pth"))
            )

        inf_source.change(
            fn=update_inf_inputs,
            inputs=[inf_source],
            outputs=[inf_model_path, inf_hf_id, inf_pth_group]
        )

        inf_button.click(
            fn=classify_plant, 
            inputs=[inf_source, inf_model_path, inf_hf_id, inf_pth_file, inf_pth_arch, inf_pth_classes, inf_input_image], 
            outputs=inf_output_label
        )

    with gr.Tab("Evaluation"):
        # 1. Model Selection
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    eval_source = gr.Radio(
                        choices=["Local", "Hugging Face Hub", "Local .pth"],
                        value="Local",
                        label="Model source"
                    )
                
                    eval_model_path = gr.Dropdown(
                        label="Select local model", 
                        choices=[], 
                        value=None, 
                        filterable=False,
                        visible=True
                    )
                    eval_hf_id = gr.Textbox(
                        label="Hugging Face model ID", 
                        placeholder="e.g. microsoft/resnet-50", 
                        visible=False
                    )
                    with gr.Column(visible=False) as eval_pth_group:
                        eval_pth_file = gr.File(label="Upload .pth file", file_types=[".pth"])
                        eval_pth_classes = gr.File(label="Upload class list (txt/json)", file_types=[".txt", ".json"])
                        eval_pth_arch = gr.Textbox(
                            label="Architecture name (timm)", 
                            placeholder="e.g. resnet50, vit_base_patch16_224"
                        )

        # 2. Test Set & Run
        with gr.Column(visible=False) as eval_run_container:
            eval_test_dir = gr.Textbox(label="Test Directory (with renamed images)", value=os.path.join("Dataset-PlantCLEF-2020-Challenge", "Test"))
            eval_button = gr.Button("Run Evaluation", variant="primary")

        # 3. Results & Export (Hidden until run)
        eval_results_state = gr.State()
        
        with gr.Column(visible=False) as eval_results_container:
            with gr.Accordion("Export Results", open=False):
                with gr.Row():
                    eval_export_dir = gr.Textbox(label="Export Directory", value="Evaluation_Results")
                    eval_export_btn = gr.Button("Export")
                eval_export_status = gr.Textbox(label="Status", interactive=False)

            eval_output_text = gr.Markdown(label="Results")
            with gr.Row():
                eval_plot_tsne = gr.Plot(label="t-SNE Visualization")
                eval_plot_am = gr.Plot(label="Activation Maps (Sample)")

        # Logic
        def update_eval_inputs(source, local_path, hf_id, pth_file):
            is_local = (source == "Local")
            is_hf = (source == "Hugging Face Hub")
            is_pth = (source == "Local .pth")
            
            show_run = False
            if is_local and local_path: show_run = True
            if is_hf and hf_id: show_run = True
            if is_pth and pth_file: show_run = True

            return (
                gr.update(visible=is_local),
                gr.update(visible=is_hf),
                gr.update(visible=is_pth),
                gr.update(visible=show_run)
            )

        eval_source.change(
            fn=update_eval_inputs,
            inputs=[eval_source, eval_model_path, eval_hf_id, eval_pth_file],
            outputs=[eval_model_path, eval_hf_id, eval_pth_group, eval_run_container]
        )

        def check_model_selected(value):
            return gr.update(visible=bool(value))

        eval_model_path.change(fn=check_model_selected, inputs=[eval_model_path], outputs=[eval_run_container])
        eval_hf_id.change(fn=check_model_selected, inputs=[eval_hf_id], outputs=[eval_run_container])
        eval_pth_file.change(fn=check_model_selected, inputs=[eval_pth_file], outputs=[eval_run_container])

        eval_button.click(
            fn=evaluate_test_set,
            inputs=[eval_source, eval_model_path, eval_hf_id, eval_pth_file, eval_pth_arch, eval_pth_classes, eval_test_dir],
            outputs=[eval_output_text, eval_plot_tsne, eval_plot_am, eval_results_state]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[eval_results_container]
        )
        
        eval_export_btn.click(
            fn=save_evaluation_results,
            inputs=[eval_results_state, eval_export_dir],
            outputs=[eval_export_status]
        )

    with gr.Tab("Training metrics"):
        metrics_model_path = gr.Dropdown(label="Select model", choices=[], value=None, filterable=False)
        with gr.Column(visible=False) as inf_plots_container:
            with gr.Accordion("Save metrics", open=False):
                with gr.Column():
                    metrics_save_btn = gr.Button("Save metrics", variant="primary")
                    metrics_status = gr.Textbox(label="Status", interactive=False, lines=2)

                metrics_save_btn.click(
                    fn=save_metrics,
                    inputs=[metrics_model_path],
                    outputs=[metrics_status]
                )

            with gr.Row():
                inf_plot_loss = gr.Plot(label="Loss")
                inf_plot_acc = gr.Plot(label="Accuracy")
            with gr.Row():
                inf_plot_lr = gr.Plot(label="Learning rate")
                inf_plot_grad = gr.Plot(label="Gradient norm")
            with gr.Row():
                inf_plot_f1 = gr.Plot(label="F1 scores")
                inf_plot_prec = gr.Plot(label="Precision")
            with gr.Row():
                inf_plot_recall = gr.Plot(label="Recall")
                inf_plot_epoch = gr.Plot(label="Epoch")
            with gr.Row():
                inf_plot_runtime = gr.Plot(label="Eval runtime")
                inf_plot_sps = gr.Plot(label="Eval samples/sec")
            with gr.Row():
                inf_plot_steps_ps = gr.Plot(label="Eval steps/sec")

        inf_plots = [
            inf_plot_loss, inf_plot_acc, inf_plot_lr, inf_plot_grad, inf_plot_f1,
            inf_plot_prec, inf_plot_recall, inf_plot_epoch, inf_plot_runtime,
            inf_plot_sps, inf_plot_steps_ps
        ]
        inf_model_path.change(
            fn=show_model_charts,
            inputs=[inf_model_path],
            outputs=inf_plots + [inf_plots_container, metrics_model_path]
        )
        metrics_model_path.change(
            fn=show_model_charts,
            inputs=[metrics_model_path],
            outputs=inf_plots + [inf_plots_container, inf_model_path]
        )

    with gr.Tab("Training"):
        with gr.Accordion("AutoTrain", open=False):
            train_autotrain_path = gr.Textbox(label="Path to AutoTrain folder")
            train_launch_button = gr.Button("Launch AutoTrain")
            train_launch_log = gr.Textbox(label="Status", interactive=False, lines=2)
            
            train_launch_button.click(
                fn=launch_autotrain_ui,
                inputs=[train_autotrain_path],
                outputs=[train_launch_log]
            )

        with gr.Accordion("TensorBoard", open=False):
            tb_log_dir = gr.Textbox(label="Log directory")
            tb_venv_dir = gr.Textbox(label="Venv parent directory (folder containing 'venv')")
            tb_launch_btn = gr.Button("Launch TensorBoard")
            tb_status = gr.Textbox(label="Status", interactive=False, lines=2)

            tb_launch_btn.click(
                fn=launch_tensorboard,
                inputs=[tb_log_dir, tb_venv_dir],
                outputs=[tb_status]
            )

    with gr.Tab("Dataset preparation"):
        
        with gr.Accordion("Clean dataset names (snake case)", open=False):
            with gr.Column():
                cn_source_dir = gr.Textbox(label="Source directory")
                cn_destination_dir = gr.Textbox(label="Destination directory")
                cn_button = gr.Button("Clean and copy", variant="primary")
                cn_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            cn_button.click(
                fn=clean_dataset_names,
                inputs=[cn_source_dir, cn_destination_dir],
                outputs=[cn_status]
            )

        with gr.Accordion("Check dataset balance", open=False):
            with gr.Column():
                db_source_dir = gr.Textbox(label="Source directory")
                db_save_files = gr.Checkbox(label="Save chart and manifest", value=False)
                with gr.Column(visible=False) as db_save_paths_container:
                    db_chart_save_path = gr.Textbox(label="Chart output path")
                    db_manifest_save_path = gr.Textbox(label="Manifest output path")
                db_check_button = gr.Button("Check", variant="primary")
                db_plot = gr.Plot(label="Class distribution")
                db_status_message = gr.Textbox(label="Status", interactive=False, lines=5)

            db_save_files.change(
                fn=lambda x: gr.update(visible=x),
                inputs=db_save_files,
                outputs=db_save_paths_container
            )

            db_check_button.click(
                fn=check_dataset_balance,
                inputs=[db_source_dir, db_save_files, db_chart_save_path, db_manifest_save_path],
                outputs=[db_plot, db_status_message]
            )

        with gr.Accordion("Check dataset splittability", open=False):
            with gr.Column():
                dss_source_dir = gr.Textbox(label="Source directory")
                dss_split_type = gr.Radio(["Train/Validate", "Train/Validate/Test"], label="Split type", value="Train/Validate")
                with gr.Row():
                    dss_train_ratio = gr.Slider(0, 100, value=80, step=1, label="Train %")
                    dss_val_ratio = gr.Slider(0, 100, value=20, step=1, label="Validate %", interactive=False)
                    dss_test_ratio = gr.Slider(0, 100, value=0, step=1, label="Test %", visible=False)
                dss_check_button = gr.Button("Check", variant="primary")
                dss_status_message = gr.Textbox(label="Status", interactive=False, lines=10)

            def dss_update_split_type(split_type):
                is_test_visible = 'Test' in split_type
                if is_test_visible:
                    # Set default ratios for Train/Validate/Test
                    return gr.update(visible=True), gr.update(value=80), gr.update(value=10), gr.update(value=10)
                else:
                    # Set default ratios for Train/Validate
                    return gr.update(visible=False), gr.update(value=80), gr.update(value=20), gr.update(value=0)

            def dss_update_ratios_from_train(train_r, test_r):
                if train_r + test_r > 100:
                    test_r = 100 - train_r
                val_r = 100 - train_r - test_r
                return gr.update(value=val_r), gr.update(value=test_r)

            def dss_update_ratios_from_test(train_r, test_r):
                if train_r + test_r > 100:
                    train_r = 100 - test_r
                val_r = 100 - train_r - test_r
                return gr.update(value=val_r), gr.update(value=train_r)

            dss_split_type.change(
                fn=dss_update_split_type,
                inputs=dss_split_type,
                outputs=[dss_test_ratio, dss_train_ratio, dss_val_ratio, dss_test_ratio]
            )
            dss_train_ratio.input(fn=dss_update_ratios_from_train, inputs=[dss_train_ratio, dss_test_ratio], outputs=[dss_val_ratio, dss_test_ratio])
            dss_test_ratio.input(fn=dss_update_ratios_from_test, inputs=[dss_train_ratio, dss_test_ratio], outputs=[dss_val_ratio, dss_train_ratio])

            dss_check_button.click(
                fn=check_dataset_splittability,
                inputs=[dss_source_dir, dss_split_type, dss_train_ratio, dss_val_ratio, dss_test_ratio],
                outputs=dss_status_message
            )

        with gr.Accordion("Split dataset", open=False):
            with gr.Column():
                ds_source_dir = gr.Textbox(label="Source directory")
                with gr.Row():
                    ds_train_output_dir = gr.Textbox(label="Train zip output path")
                    ds_val_output_dir = gr.Textbox(label="Validate zip output path")
                    ds_test_output_dir = gr.Textbox(label="Test zip output path", visible=False)
                with gr.Row():
                    ds_train_manifest_path = gr.Textbox(label="Train manifest output path")
                    ds_val_manifest_path = gr.Textbox(label="Validate manifest output path")
                    ds_test_manifest_path = gr.Textbox(label="Test manifest output path", visible=False)
                ds_split_type = gr.Radio(["Train/Validate", "Train/Validate/Test"], label="Split type", value="Train/Validate")
                with gr.Row():
                    ds_train_ratio = gr.Slider(0, 100, value=80, step=1, label="Train %")
                    ds_val_ratio = gr.Slider(0, 100, value=20, step=1, label="Validate %", interactive=False)
                    ds_test_ratio = gr.Slider(0, 100, value=0, step=1, label="Test %", visible=False)
                ds_split_button = gr.Button("Split", variant="primary")
                ds_status_message = gr.Textbox(label="Status", interactive=False, lines=5)

            def update_split_type(split_type):
                is_test_visible = 'Test' in split_type
                if is_test_visible:
                    # Set default ratios for Train/Validate/Test
                    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(value=80), gr.update(value=10), gr.update(value=10)
                else:
                    # Set default ratios for Train/Validate
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=80), gr.update(value=20), gr.update(value=0)

            def update_ratios_from_train(train_r, test_r):
                if train_r + test_r > 100:
                    test_r = 100 - train_r
                val_r = 100 - train_r - test_r
                return gr.update(value=val_r), gr.update(value=test_r)

            def update_ratios_from_test(train_r, test_r):
                if train_r + test_r > 100:
                    train_r = 100 - test_r
                val_r = 100 - train_r - test_r
                return gr.update(value=val_r), gr.update(value=train_r)

            ds_split_type.change(
                fn=update_split_type,
                inputs=ds_split_type,
                outputs=[ds_test_ratio, ds_test_output_dir, ds_test_manifest_path, ds_train_ratio, ds_val_ratio, ds_test_ratio]
            )
            ds_train_ratio.input(fn=update_ratios_from_train, inputs=[ds_train_ratio, ds_test_ratio], outputs=[ds_val_ratio, ds_test_ratio])
            ds_test_ratio.input(fn=update_ratios_from_test, inputs=[ds_train_ratio, ds_test_ratio], outputs=[ds_val_ratio, ds_train_ratio])

            ds_split_button.click(
                fn=split_dataset,
                inputs=[ds_source_dir, ds_train_output_dir, ds_val_output_dir, ds_test_output_dir, ds_train_manifest_path, ds_val_manifest_path, ds_test_manifest_path, ds_split_type, ds_train_ratio, ds_val_ratio, ds_test_ratio],
                outputs=ds_status_message
            )

    with gr.Tab("Utilities"):
        with gr.Accordion("Generate directory manifest", open=False):
            with gr.Column():
                dp_directory_path = gr.Textbox(
                    label="Directory path"
                )
                dp_manifest_save_path = gr.Textbox(
                    label="Manifest output path"
                )
                dp_manifest_type = gr.Radio(["Directories only", "Directories and files"], label="Manifest content", value="Directories only")
                dp_generate_button = gr.Button("Generate", variant="primary")
                dp_status_message = gr.Textbox(label="Status", interactive=False, lines=5)
            
            dp_generate_button.click(
                fn=generate_manifest,
                inputs=[dp_directory_path, dp_manifest_save_path, dp_manifest_type],
                outputs=[dp_status_message]
            )

    with gr.Tab("Custom", visible=CUSTOM_UTILS_AVAILABLE):
        with gr.Accordion("Evaluation (Dataset - MRR & t-SNE)", open=False):
            with gr.Column():
                eval_ds_model_path = gr.Dropdown(label="Select Model", choices=[], value=None)
                eval_ds_dir = gr.Textbox(label="Dataset Directory (Class folders)", value="")
                eval_ds_button = gr.Button("Run Evaluation", variant="primary")
                eval_ds_output_text = gr.Textbox(label="Results", interactive=False)
                eval_ds_plot = gr.Plot(label="t-SNE Visualization")
            
            eval_ds_button.click(
                fn=evaluate_dataset,
                inputs=[eval_ds_model_path, eval_ds_dir],
                outputs=[eval_ds_output_text, eval_ds_plot]
            )

        with gr.Accordion("Sort Dataset (PlantCLEF)", open=False):
            with gr.Column():
                cust_source_dir = gr.Textbox(label="Source directory")
                cust_destination_dir = gr.Textbox(label="Destination directory")
                cust_species_list_path = gr.Textbox(label="Species List Path (ID;Name format)")
                cust_pairs_list_path = gr.Textbox(label="Pairs List Path (Optional, list of IDs)")
                cust_sort_button = gr.Button("Sort Dataset", variant="primary")
                cust_status_message = gr.Textbox(label="Status", interactive=False, lines=5)

            cust_sort_button.click(
                fn=custom_sort_dataset,
                inputs=[cust_source_dir, cust_destination_dir, cust_species_list_path, cust_pairs_list_path],
                outputs=[cust_status_message]
            )

        with gr.Accordion("Rename Test Images (PlantCLEF)", open=False):
            with gr.Column():
                rti_test_dir = gr.Textbox(label="Test Directory", value=os.path.join("Dataset-PlantCLEF-2020-Challenge", "Test"))
                rti_groundtruth_path = gr.Textbox(label="Groundtruth File Path", value=os.path.join("AML-dataset", "AML_project_herbarium_dataset", "list", "groundtruth.txt"))
                rti_species_list_path = gr.Textbox(label="Species List Path", value=os.path.join("AML-dataset", "AML_project_herbarium_dataset", "list", "species_list.txt"))
                rti_button = gr.Button("Rename Images", variant="primary")
                rti_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            rti_button.click(
                fn=rename_test_images_func,
                inputs=[rti_test_dir, rti_groundtruth_path, rti_species_list_path],
                outputs=[rti_status]
            )

    refresh_button.click(
        fn=update_model_choices,
        inputs=[inf_model_path],
        outputs=[inf_model_path, metrics_model_path, eval_model_path, eval_ds_model_path]
    )
    demo.load(
        fn=update_model_choices,
        inputs=[],
        outputs=[inf_model_path, metrics_model_path, eval_model_path, eval_ds_model_path]
    )

if __name__ == "__main__":
    demo.launch()
