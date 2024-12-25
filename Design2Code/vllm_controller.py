import subprocess, os
def start_vllm_server(log_file, port=18999, model="Qwen/Qwen2-VL-7B-Instruct", is_llm=False, cuda_visible_devices="0"):
    try:
        # Set the CUDA_VISIBLE_DEVICES environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        # Start the VLLM server process
        if is_llm:
            command = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model,
                "--trust-remote-code",
                "--dtype", "auto",
                "--api-key", "token-abc123s", "--max-model-len", "32768",
                "--port", f"{port}"
            ]
        else:
            command = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model,
                "--trust-remote-code",
                "--dtype", "auto",
                "--api-key", "token-abc123s", "--max-model-len", "32768",
                "--port", f"{port}", "--limit-mm-per-prompt", "image=5"
            ]

        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

        print("VLLM server process started successfully.")
        return process
    except Exception as e:
        print(f"Failed to start VLLM serve: {e}")
        return None

def stop_vllm_server(process):
    try:
        if process:
            process.terminate()  # Gracefully terminate the process
            process.wait()  # Wait for it to finish
            print("VLLM server process stopped.")
        else:
            print("No process to stop.")
    except Exception as e:
        print(f"Failed to stop VLLM serve: {e}")