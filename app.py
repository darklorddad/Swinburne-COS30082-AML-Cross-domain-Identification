import os
import json
import gradio as gr
from gradio_wrapper import (
    classify_plant, show_model_charts, get_model_choices, update_model_choices,
    launch_autotrain_ui, launch_tensorboard, generate_manifest,
    split_dataset, check_dataset_balance, check_dataset_splittability,
    clean_dataset_names, save_metrics, evaluate_test_set, save_evaluation_results,
    get_placeholder_plot
)
try:
    from custom_utils import custom_sort_dataset, rename_test_images_func, sort_test_dataset, separate_paired_species, split_paired_dataset_custom, split_hybrid_dataset
    CUSTOM_UTILS_AVAILABLE = True
except ImportError:
    CUSTOM_UTILS_AVAILABLE = False
    def custom_sort_dataset(*args, **kwargs): raise gr.Error("custom_utils not available")
    def rename_test_images_func(*args, **kwargs): raise gr.Error("custom_utils not available")
    def sort_test_dataset(*args, **kwargs): raise gr.Error("custom_utils not available")
    def separate_paired_species(*args, **kwargs): raise gr.Error("custom_utils not available")
    def split_paired_dataset_custom(*args, **kwargs): raise gr.Error("custom_utils not available")
    def split_hybrid_dataset(*args, **kwargs): raise gr.Error("custom_utils not available")

DEFAULT_MANIFEST_PATH = os.path.join('core', 'manifest.md').replace(os.sep, '/')

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_setting(key, value):
    config = load_config()
    config[key] = value
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

app_config = load_config()

# #############################################################################
# GRADIO UI
# #############################################################################

with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}") as demo:

    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    inf_source = gr.Radio(
                        choices=["Local", "Hugging Face hub", "Local .pth"],
                        value="Local",
                        label="Model source"
                    )
                    
                    inf_model_path = gr.Dropdown(
                        label="Select local model", 
                        choices=[], 
                        value=None, 
                        filterable=False,
                        visible=True,
                        allow_custom_value=False
                    )
                    
                    inf_hf_id = gr.Textbox(
                        label="Hugging Face model ID", 
                        visible=False,
                        value=app_config.get("inf_hf_id", "")
                    )
                    inf_hf_id.change(lambda x: save_setting("inf_hf_id", x), inputs=[inf_hf_id])
                    
                    with gr.Column(visible=False) as inf_pth_group:
                        inf_pth_file = gr.Textbox(label="Path to .pth file", value=app_config.get("inf_pth_file", ""))
                        inf_pth_file.change(lambda x: save_setting("inf_pth_file", x), inputs=[inf_pth_file])

                        inf_pth_classes = gr.Textbox(label="Path to class list (txt/json)", value=app_config.get("inf_pth_classes", ""))
                        inf_pth_classes.change(lambda x: save_setting("inf_pth_classes", x), inputs=[inf_pth_classes])

                        inf_pth_arch = gr.Textbox(
                            label="Architecture name (timm)",
                            value=app_config.get("inf_pth_arch", "")
                        )
                        inf_pth_arch.change(lambda x: save_setting("inf_pth_arch", x), inputs=[inf_pth_arch])

                inf_input_image = gr.Image(type="pil", label="Upload an image")

            with gr.Column(scale=1):
                inf_output_label = gr.Label(num_top_classes=5, label="Predictions")
                inf_heatmap = gr.Image(label="Heatmap")
                inf_button = gr.Button("Classify", variant="primary")

        def update_inf_inputs(source):
            return (
                gr.update(visible=(source == "Local")),
                gr.update(visible=(source == "Hugging Face hub")),
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
            outputs=[inf_output_label, inf_heatmap]
        )

    with gr.Tab("Evaluation"):
        # 1. Model Selection
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    eval_source = gr.Radio(
                        choices=["Local", "Hugging Face hub", "Local .pth"],
                        value="Local",
                        label="Model source"
                    )
                
                    eval_model_path = gr.Dropdown(
                        label="Select local model", 
                        choices=[], 
                        value=None, 
                        filterable=False,
                        visible=True,
                        allow_custom_value=False
                    )
                    eval_hf_id = gr.Textbox(
                        label="Hugging Face model ID", 
                        visible=False,
                        value=app_config.get("eval_hf_id", "")
                    )
                    eval_hf_id.change(lambda x: save_setting("eval_hf_id", x), inputs=[eval_hf_id])

                    with gr.Column(visible=False) as eval_pth_group:
                        eval_pth_file = gr.Textbox(label="Path to .pth file", value=app_config.get("eval_pth_file", ""))
                        eval_pth_file.change(lambda x: save_setting("eval_pth_file", x), inputs=[eval_pth_file])

                        eval_pth_classes = gr.Textbox(label="Path to class list (txt/json)", value=app_config.get("eval_pth_classes", ""))
                        eval_pth_classes.change(lambda x: save_setting("eval_pth_classes", x), inputs=[eval_pth_classes])

                        eval_pth_arch = gr.Textbox(
                            label="Architecture name (timm)",
                            value=app_config.get("eval_pth_arch", "")
                        )
                        eval_pth_arch.change(lambda x: save_setting("eval_pth_arch", x), inputs=[eval_pth_arch])

        # 2. Test Set & Run
        with gr.Column(visible=True) as eval_run_container:
            with gr.Accordion("Settings", open=False):
                eval_test_dir = gr.Textbox(label="Path to test set", value=app_config.get("eval_test_dir", os.path.join("Dataset-PlantCLEF-2020-Challenge", "Images", "Test-set")))
                eval_test_dir.change(lambda x: save_setting("eval_test_dir", x), inputs=[eval_test_dir])

                eval_mode = gr.Radio(["Standard", "Prototype retrieval"], label="Evaluation mode", value="Standard")
                
                eval_ref_dir = gr.Textbox(label="Path to reference set (for prototypes)", visible=False, value=app_config.get("eval_ref_dir", ""))
                eval_ref_dir.change(lambda x: save_setting("eval_ref_dir", x), inputs=[eval_ref_dir])

                eval_batch_size = gr.Slider(minimum=1, maximum=128, value=32, step=1, label="Batch size")
                eval_perplexity = gr.Slider(minimum=2, maximum=100, value=30, step=1, label="t-SNE perplexity")
            eval_button = gr.Button("Run evaluation", variant="primary")

        # 3. Save Evaluation (Hidden until run)
        with gr.Column(visible=False) as eval_save_container:
            with gr.Accordion("Save evaluation", open=False):
                with gr.Column():
                    eval_export_dir = gr.Textbox(label="Export directory", value=app_config.get("eval_export_dir", ""))
                    eval_export_dir.change(lambda x: save_setting("eval_export_dir", x), inputs=[eval_export_dir])

                    eval_export_btn = gr.Button("Save", variant="primary")
                    eval_export_status = gr.Textbox(label="Status", interactive=False)

        # 4. Results (Hidden until run)
        eval_results_state = gr.State()
        
        with gr.Column(visible=False) as eval_results_container:
            with gr.Row():
                eval_plot_tsne = gr.Plot(label="t-SNE visualisation", value=get_placeholder_plot())
                eval_plot_metrics = gr.Plot(label="Metrics", value=get_placeholder_plot())

        # Logic
        def update_eval_inputs(source):
            is_local = (source == "Local")
            is_hf = (source == "Hugging Face hub")
            is_pth = (source == "Local .pth")
            
            return (
                gr.update(visible=is_local),
                gr.update(visible=is_hf),
                gr.update(visible=is_pth)
            )

        eval_source.change(
            fn=update_eval_inputs,
            inputs=[eval_source],
            outputs=[eval_model_path, eval_hf_id, eval_pth_group]
        )

        def update_eval_mode(mode):
            return gr.update(visible=(mode == "Prototype retrieval"))

        eval_mode.change(fn=update_eval_mode, inputs=[eval_mode], outputs=[eval_ref_dir])

        eval_button.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[eval_results_container, eval_save_container]
        ).then(
            fn=evaluate_test_set,
            inputs=[eval_source, eval_model_path, eval_hf_id, eval_pth_file, eval_pth_arch, eval_pth_classes, eval_test_dir, eval_batch_size, eval_perplexity, eval_mode, eval_ref_dir],
            outputs=[eval_plot_tsne, eval_plot_metrics, eval_results_state]
        )
        
        eval_export_btn.click(
            fn=save_evaluation_results,
            inputs=[eval_results_state, eval_export_dir],
            outputs=[eval_export_status]
        )

    with gr.Tab("Training metrics"):
        metrics_model_path = gr.Dropdown(label="Select local model", choices=[], value=None, filterable=False, allow_custom_value=False)
        metrics_load_btn = gr.Button("Load metrics", variant="primary")

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
                inf_plot_loss = gr.Plot(label="Loss", value=get_placeholder_plot())
                inf_plot_acc = gr.Plot(label="Accuracy", value=get_placeholder_plot())
            with gr.Row():
                inf_plot_lr = gr.Plot(label="Learning rate", value=get_placeholder_plot())
                inf_plot_grad = gr.Plot(label="Gradient norm", value=get_placeholder_plot())
            with gr.Row():
                inf_plot_f1 = gr.Plot(label="F1 scores", value=get_placeholder_plot())
                inf_plot_prec = gr.Plot(label="Precision", value=get_placeholder_plot())
            with gr.Row():
                inf_plot_recall = gr.Plot(label="Recall", value=get_placeholder_plot())
                inf_plot_epoch = gr.Plot(label="Epoch", value=get_placeholder_plot())
            with gr.Row():
                inf_plot_runtime = gr.Plot(label="Eval runtime", value=get_placeholder_plot())
                inf_plot_sps = gr.Plot(label="Eval samples/sec", value=get_placeholder_plot())
            with gr.Row():
                inf_plot_steps_ps = gr.Plot(label="Eval steps/sec", value=get_placeholder_plot())

        inf_plots = [
            inf_plot_loss, inf_plot_acc, inf_plot_lr, inf_plot_grad, inf_plot_f1,
            inf_plot_prec, inf_plot_recall, inf_plot_epoch, inf_plot_runtime,
            inf_plot_sps, inf_plot_steps_ps
        ]
        metrics_load_btn.click(
            fn=lambda: [get_placeholder_plot()] * 11 + [gr.update(visible=True)],
            outputs=inf_plots + [inf_plots_container]
        ).then(
            fn=show_model_charts,
            inputs=[metrics_model_path],
            outputs=inf_plots + [inf_plots_container]
        )

    with gr.Tab("Training"):
        with gr.Accordion("AutoTrain", open=False):
            train_autotrain_path = gr.Textbox(label="Path to AutoTrain folder", value=app_config.get("train_autotrain_path", ""))
            train_autotrain_path.change(lambda x: save_setting("train_autotrain_path", x), inputs=[train_autotrain_path])

            train_launch_button = gr.Button("Launch AutoTrain")
            train_launch_log = gr.Textbox(label="Status", interactive=False, lines=2)
            
            train_launch_button.click(
                fn=launch_autotrain_ui,
                inputs=[train_autotrain_path],
                outputs=[train_launch_log]
            )

        with gr.Accordion("TensorBoard", open=False):
            tb_log_dir = gr.Textbox(label="Log directory", value=app_config.get("tb_log_dir", ""))
            tb_log_dir.change(lambda x: save_setting("tb_log_dir", x), inputs=[tb_log_dir])

            tb_venv_dir = gr.Textbox(label="Venv parent directory (folder containing 'venv')", value=app_config.get("tb_venv_dir", ""))
            tb_venv_dir.change(lambda x: save_setting("tb_venv_dir", x), inputs=[tb_venv_dir])

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
                cn_source_dir = gr.Textbox(label="Source directory", value=app_config.get("cn_source_dir", ""))
                cn_source_dir.change(lambda x: save_setting("cn_source_dir", x), inputs=[cn_source_dir])

                cn_destination_dir = gr.Textbox(label="Destination directory", value=app_config.get("cn_destination_dir", ""))
                cn_destination_dir.change(lambda x: save_setting("cn_destination_dir", x), inputs=[cn_destination_dir])

                cn_button = gr.Button("Clean and copy", variant="primary")
                cn_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            cn_button.click(
                fn=clean_dataset_names,
                inputs=[cn_source_dir, cn_destination_dir],
                outputs=[cn_status]
            )

        with gr.Accordion("Check dataset balance", open=False):
            with gr.Column():
                db_source_dir = gr.Textbox(label="Source directory", value=app_config.get("db_source_dir", ""))
                db_source_dir.change(lambda x: save_setting("db_source_dir", x), inputs=[db_source_dir])

                db_save_files = gr.Checkbox(label="Save chart and manifest", value=False)
                with gr.Column(visible=False) as db_save_paths_container:
                    db_chart_save_path = gr.Textbox(label="Chart output path", value=app_config.get("db_chart_save_path", ""))
                    db_chart_save_path.change(lambda x: save_setting("db_chart_save_path", x), inputs=[db_chart_save_path])

                    db_manifest_save_path = gr.Textbox(label="Manifest output path", value=app_config.get("db_manifest_save_path", ""))
                    db_manifest_save_path.change(lambda x: save_setting("db_manifest_save_path", x), inputs=[db_manifest_save_path])

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
                dss_source_dir = gr.Textbox(label="Source directory", value=app_config.get("dss_source_dir", ""))
                dss_source_dir.change(lambda x: save_setting("dss_source_dir", x), inputs=[dss_source_dir])

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
                ds_source_dir = gr.Textbox(label="Source directory", value=app_config.get("ds_source_dir", ""))
                ds_source_dir.change(lambda x: save_setting("ds_source_dir", x), inputs=[ds_source_dir])

                with gr.Row():
                    ds_train_output_dir = gr.Textbox(label="Train zip output path", value=app_config.get("ds_train_output_dir", ""))
                    ds_train_output_dir.change(lambda x: save_setting("ds_train_output_dir", x), inputs=[ds_train_output_dir])

                    ds_val_output_dir = gr.Textbox(label="Validate zip output path", value=app_config.get("ds_val_output_dir", ""))
                    ds_val_output_dir.change(lambda x: save_setting("ds_val_output_dir", x), inputs=[ds_val_output_dir])

                    ds_test_output_dir = gr.Textbox(label="Test zip output path", visible=False, value=app_config.get("ds_test_output_dir", ""))
                    ds_test_output_dir.change(lambda x: save_setting("ds_test_output_dir", x), inputs=[ds_test_output_dir])

                with gr.Row():
                    ds_train_manifest_path = gr.Textbox(label="Train manifest output path", value=app_config.get("ds_train_manifest_path", ""))
                    ds_train_manifest_path.change(lambda x: save_setting("ds_train_manifest_path", x), inputs=[ds_train_manifest_path])

                    ds_val_manifest_path = gr.Textbox(label="Validate manifest output path", value=app_config.get("ds_val_manifest_path", ""))
                    ds_val_manifest_path.change(lambda x: save_setting("ds_val_manifest_path", x), inputs=[ds_val_manifest_path])

                    ds_test_manifest_path = gr.Textbox(label="Test manifest output path", visible=False, value=app_config.get("ds_test_manifest_path", ""))
                    ds_test_manifest_path.change(lambda x: save_setting("ds_test_manifest_path", x), inputs=[ds_test_manifest_path])
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
                    label="Directory path",
                    value=app_config.get("dp_directory_path", "")
                )
                dp_directory_path.change(lambda x: save_setting("dp_directory_path", x), inputs=[dp_directory_path])

                dp_manifest_save_path = gr.Textbox(
                    label="Manifest output path",
                    value=app_config.get("dp_manifest_save_path", "")
                )
                dp_manifest_save_path.change(lambda x: save_setting("dp_manifest_save_path", x), inputs=[dp_manifest_save_path])
                dp_manifest_type = gr.Radio(["Directories only", "Directories and files"], label="Manifest content", value="Directories only")
                dp_generate_button = gr.Button("Generate", variant="primary")
                dp_status_message = gr.Textbox(label="Status", interactive=False, lines=5)
            
            dp_generate_button.click(
                fn=generate_manifest,
                inputs=[dp_directory_path, dp_manifest_save_path, dp_manifest_type],
                outputs=[dp_status_message]
            )

    with gr.Tab("Custom", visible=CUSTOM_UTILS_AVAILABLE):
        with gr.Accordion("Sort Test Dataset (PlantCLEF)", open=False):
            with gr.Column():
                std_test_dir = gr.Textbox(label="Test Directory (Flat images)", value=app_config.get("std_test_dir", ""))
                std_test_dir.change(lambda x: save_setting("std_test_dir", x), inputs=[std_test_dir])

                std_dest_dir = gr.Textbox(label="Destination Directory (Class folders)", value=app_config.get("std_dest_dir", ""))
                std_dest_dir.change(lambda x: save_setting("std_dest_dir", x), inputs=[std_dest_dir])

                std_groundtruth_path = gr.Textbox(label="Groundtruth File Path", value=app_config.get("std_groundtruth_path", ""))
                std_groundtruth_path.change(lambda x: save_setting("std_groundtruth_path", x), inputs=[std_groundtruth_path])

                std_species_list_path = gr.Textbox(label="Species List Path", value=app_config.get("std_species_list_path", ""))
                std_species_list_path.change(lambda x: save_setting("std_species_list_path", x), inputs=[std_species_list_path])

                std_button = gr.Button("Sort Test Set", variant="primary")
                std_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            std_button.click(
                fn=sort_test_dataset,
                inputs=[std_test_dir, std_dest_dir, std_groundtruth_path, std_species_list_path],
                outputs=[std_status]
            )

        with gr.Accordion("Sort Dataset (PlantCLEF)", open=False):
            with gr.Column():
                cust_source_dir = gr.Textbox(label="Source directory", value=app_config.get("cust_source_dir", ""))
                cust_source_dir.change(lambda x: save_setting("cust_source_dir", x), inputs=[cust_source_dir])

                cust_destination_dir = gr.Textbox(label="Destination directory", value=app_config.get("cust_destination_dir", ""))
                cust_destination_dir.change(lambda x: save_setting("cust_destination_dir", x), inputs=[cust_destination_dir])

                cust_species_list_path = gr.Textbox(label="Species List Path (ID;Name format)", value=app_config.get("cust_species_list_path", ""))
                cust_species_list_path.change(lambda x: save_setting("cust_species_list_path", x), inputs=[cust_species_list_path])

                cust_pairs_list_path = gr.Textbox(label="Pairs List Path (Optional, list of IDs)", value=app_config.get("cust_pairs_list_path", ""))
                cust_pairs_list_path.change(lambda x: save_setting("cust_pairs_list_path", x), inputs=[cust_pairs_list_path])

                cust_sort_button = gr.Button("Sort Dataset", variant="primary")
                cust_status_message = gr.Textbox(label="Status", interactive=False, lines=5)

            cust_sort_button.click(
                fn=custom_sort_dataset,
                inputs=[cust_source_dir, cust_destination_dir, cust_species_list_path, cust_pairs_list_path],
                outputs=[cust_status_message]
            )

        with gr.Accordion("Rename Test Images (PlantCLEF)", open=False):
            with gr.Column():
                rti_test_dir = gr.Textbox(label="Test Directory", value=app_config.get("rti_test_dir", ""))
                rti_test_dir.change(lambda x: save_setting("rti_test_dir", x), inputs=[rti_test_dir])

                rti_groundtruth_path = gr.Textbox(label="Groundtruth File Path", value=app_config.get("rti_groundtruth_path", ""))
                rti_groundtruth_path.change(lambda x: save_setting("rti_groundtruth_path", x), inputs=[rti_groundtruth_path])

                rti_species_list_path = gr.Textbox(label="Species List Path", value=app_config.get("rti_species_list_path", ""))
                rti_species_list_path.change(lambda x: save_setting("rti_species_list_path", x), inputs=[rti_species_list_path])

                rti_button = gr.Button("Rename Images", variant="primary")
                rti_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            rti_button.click(
                fn=rename_test_images_func,
                inputs=[rti_test_dir, rti_groundtruth_path, rti_species_list_path],
                outputs=[rti_status]
            )

        with gr.Accordion("Separate Paired/Unpaired Species", open=False):
            with gr.Column():
                sps_source_dir = gr.Textbox(label="Source Directory (Species folders)", value=app_config.get("sps_source_dir", ""))
                sps_source_dir.change(lambda x: save_setting("sps_source_dir", x), inputs=[sps_source_dir])

                sps_output_dir = gr.Textbox(label="Output Directory", value=app_config.get("sps_output_dir", ""))
                sps_output_dir.change(lambda x: save_setting("sps_output_dir", x), inputs=[sps_output_dir])

                sps_button = gr.Button("Separate Dataset", variant="primary")
                sps_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            sps_button.click(
                fn=separate_paired_species,
                inputs=[sps_source_dir, sps_output_dir],
                outputs=[sps_status]
            )

        with gr.Accordion("Split Paired Dataset (Val = Photos only)", open=False):
            with gr.Column():
                spd_source_dir = gr.Textbox(label="Source Directory (Paired species folders)", value=app_config.get("spd_source_dir", ""))
                spd_source_dir.change(lambda x: save_setting("spd_source_dir", x), inputs=[spd_source_dir])

                spd_output_dir = gr.Textbox(label="Output Directory", value=app_config.get("spd_output_dir", ""))
                spd_output_dir.change(lambda x: save_setting("spd_output_dir", x), inputs=[spd_output_dir])
                
                with gr.Row():
                    spd_val_ratio = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Validation Ratio (%) (Photos only)")
                    spd_min_items = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Minimum items per set")

                spd_button = gr.Button("Split Dataset", variant="primary")
                spd_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            spd_button.click(
                fn=split_paired_dataset_custom,
                inputs=[spd_source_dir, spd_output_dir, spd_val_ratio, spd_min_items],
                outputs=[spd_status]
            )

        with gr.Accordion("Split Hybrid Dataset (Prioritise Photos for Val)", open=False):
            with gr.Column():
                shd_source_dir = gr.Textbox(label="Source Directory (Mixed species folders)", value=app_config.get("shd_source_dir", ""))
                shd_source_dir.change(lambda x: save_setting("shd_source_dir", x), inputs=[shd_source_dir])

                shd_output_dir = gr.Textbox(label="Output Directory", value=app_config.get("shd_output_dir", ""))
                shd_output_dir.change(lambda x: save_setting("shd_output_dir", x), inputs=[shd_output_dir])
                
                with gr.Row():
                    shd_val_ratio = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Validation Ratio (%)")
                    shd_min_items = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Minimum items per set")

                shd_button = gr.Button("Split Hybrid Dataset", variant="primary")
                shd_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            shd_button.click(
                fn=split_hybrid_dataset,
                inputs=[shd_source_dir, shd_output_dir, shd_val_ratio, shd_min_items],
                outputs=[shd_status]
            )

    def load_saved_settings():
        config = load_config()
        return [
            config.get("inf_hf_id", ""),
            config.get("inf_pth_file", ""),
            config.get("inf_pth_classes", ""),
            config.get("inf_pth_arch", ""),
            config.get("eval_hf_id", ""),
            config.get("eval_pth_file", ""),
            config.get("eval_pth_classes", ""),
            config.get("eval_pth_arch", ""),
            config.get("eval_test_dir", os.path.join("Dataset-PlantCLEF-2020-Challenge", "Images", "Test-set")),
            config.get("eval_ref_dir", ""),
            config.get("eval_export_dir", ""),
            config.get("train_autotrain_path", ""),
            config.get("tb_log_dir", ""),
            config.get("tb_venv_dir", ""),
            config.get("cn_source_dir", ""),
            config.get("cn_destination_dir", ""),
            config.get("db_source_dir", ""),
            config.get("db_chart_save_path", ""),
            config.get("db_manifest_save_path", ""),
            config.get("dss_source_dir", ""),
            config.get("ds_source_dir", ""),
            config.get("ds_train_output_dir", ""),
            config.get("ds_val_output_dir", ""),
            config.get("ds_test_output_dir", ""),
            config.get("ds_train_manifest_path", ""),
            config.get("ds_val_manifest_path", ""),
            config.get("ds_test_manifest_path", ""),
            config.get("dp_directory_path", ""),
            config.get("dp_manifest_save_path", ""),
            config.get("std_test_dir", ""),
            config.get("std_dest_dir", ""),
            config.get("std_groundtruth_path", ""),
            config.get("std_species_list_path", ""),
            config.get("cust_source_dir", ""),
            config.get("cust_destination_dir", ""),
            config.get("cust_species_list_path", ""),
            config.get("cust_pairs_list_path", ""),
            config.get("rti_test_dir", ""),
            config.get("rti_groundtruth_path", ""),
            config.get("rti_species_list_path", ""),
            config.get("sps_source_dir", ""),
            config.get("sps_output_dir", ""),
            config.get("spd_source_dir", ""),
            config.get("spd_output_dir", ""),
            config.get("shd_source_dir", ""),
            config.get("shd_output_dir", "")
        ]

    demo.load(
        fn=load_saved_settings,
        inputs=[],
        outputs=[
            inf_hf_id, inf_pth_file, inf_pth_classes, inf_pth_arch,
            eval_hf_id, eval_pth_file, eval_pth_classes, eval_pth_arch,
            eval_test_dir, eval_ref_dir, eval_export_dir,
            train_autotrain_path, tb_log_dir, tb_venv_dir,
            cn_source_dir, cn_destination_dir,
            db_source_dir, db_chart_save_path, db_manifest_save_path,
            dss_source_dir,
            ds_source_dir, ds_train_output_dir, ds_val_output_dir, ds_test_output_dir,
            ds_train_manifest_path, ds_val_manifest_path, ds_test_manifest_path,
            dp_directory_path, dp_manifest_save_path,
            std_test_dir, std_dest_dir, std_groundtruth_path, std_species_list_path,
            cust_source_dir, cust_destination_dir, cust_species_list_path, cust_pairs_list_path,
            rti_test_dir, rti_groundtruth_path, rti_species_list_path,
            sps_source_dir, sps_output_dir,
            spd_source_dir, spd_output_dir,
            shd_source_dir, shd_output_dir
        ]
    )

    demo.load(fn=lambda: update_model_choices("inference"), outputs=[inf_model_path])
    demo.load(fn=lambda: update_model_choices("metrics"), outputs=[metrics_model_path])
    demo.load(fn=lambda: update_model_choices("evaluation"), outputs=[eval_model_path])

if __name__ == "__main__":
    demo.launch()
