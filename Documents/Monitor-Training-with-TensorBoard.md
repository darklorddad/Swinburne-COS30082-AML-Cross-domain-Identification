# How to Monitor Training with TensorBoard

This guide explains how to visualize machine learning training metrics (loss, accuracy, learning rate) in real-time using TensorBoard.

## 1. Prerequisites
Ensure TensorBoard is installed in your Python environment.

```bash
pip install tensorboard
```

## 2. How to Run TensorBoard

**Note:** Do not close your training script. TensorBoard runs as a separate process in a separate terminal window.

### Step 1: Open a New Terminal
Open a new Command Prompt or PowerShell window.

### Step 2: Activate Your Virtual Environment
You must be inside the same Python environment where you installed TensorBoard. Run the command corresponding to your system:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
```

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

**Anaconda (Conda):**
```bash
conda activate your_env_name
```

### Step 3: Locate Your Log Directory
Find the folder where your model is saving the logs.
*   This folder will contain a file starting with `events.out.tfevents...`.
*   Common default folder names are `runs`, `logs`, `lightning_logs`, or your **Project Name**.

### Step 4: Run the Command
Run the following command. Replace `"path/to/folder"` with the actual path to your logs.

**Relative Path (if you are in the project root):**
```bash
tensorboard --logdir "Model-Logs-Folder"
```

**Absolute Path (failsafe method):**
```bash
tensorboard --logdir "C:\Users\Name\Project\Model-Logs-Folder"
```

### Step 5: View the Dashboard
1.  The terminal will output a URL (usually `http://localhost:6006/`).
2.  Open your web browser (Chrome, Edge, Firefox).
3.  Navigate to **[http://localhost:6006/](http://localhost:6006/)**.

The page will automatically refresh periodically to show the latest training data.

---

## 3. Common Commands & Troubleshooting

### "Port 6006 is already in use"
If you have another TensorBoard instance running, or the port is blocked, run on a specific port (e.g., 6007):
```bash
tensorboard --logdir "path/to/folder" --port 6007
```

### "No dashboards are active for the current data set"
If the web page is empty:
1.  **Check the Path:** Ensure the `--logdir` points to the folder *containing* the `runs` folder or the `events.out` file.
2.  **Wait:** If training just started, wait 1–2 minutes. The model may not have written the first log file yet.
3.  **Reload:** Press the refresh button in the top right corner of the TensorBoard interface.

### Specify Reload Interval
To make TensorBoard refresh data faster (default is every 30 seconds):
```bash
tensorboard --logdir "path/to/folder" --reload_interval 5
```