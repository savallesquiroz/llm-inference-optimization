"""
Comprehensive LLM Inference Optimization Benchmark Suite
Run this script to automate all optimization experiments
"""

import subprocess
import csv
import time
import re
import os
import json
import signal
from pathlib import Path
from datetime import datetime


class LLMBenchmarkSuite:
    def __init__(self, llama_cpp_path, models_path, output_dir="results"):
        self.llama_cpp_path = Path(llama_cpp_path)
        self.models_path = Path(models_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Standard test prompt
        self.test_prompt = "The quick brown fox jumps over the lazy dog."

        # Locate the built llama-cli executable in common locations
        self.executable = self._find_executable()
        if self.executable is None:
            print("⚠️ Warning: could not find llama-cli executable under", self.llama_cpp_path)
            print("Please set LLAMA_CPP_PATH to the folder containing the built 'llama-cli.exe' or place the exe in the repo root.")

    def _find_executable(self):
        candidates = [
            self.llama_cpp_path / 'build' / 'bin' / 'Release' / 'llama-cli.exe',
            self.llama_cpp_path / 'build' / 'bin' / 'llama-cli.exe',
            self.llama_cpp_path / 'build' / 'llama-cli.exe',
            self.llama_cpp_path / 'llama-cli.exe',
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None
        
    def _lock_path(self, model_file):
        safe = model_file.replace('/', '_').replace('\\', '_')
        locks = self.output_dir / 'locks'
        locks.mkdir(exist_ok=True)
        return locks / f"{safe}.lock"

    def _acquire_lock(self, model_file, cmd):
        lock = self._lock_path(model_file)
        if lock.exists():
            try:
                data = json.loads(lock.read_text())
                pid = data.get('pid')
                alive = False
                if pid:
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except OSError:
                        alive = False
                if alive:
                    print(f"⚠️ Another run (PID {pid}) is active for {model_file}; skipping to avoid duplicate.")
                    return False
                else:
                    print(f"Stale lock for {model_file} found; removing.")
                    lock.unlink()
            except Exception as e:
                print(f"Could not read lock file {lock}; assuming active to be safe. Error: {e}")
                return False
        payload = {'pid': os.getpid(), 'started': datetime.now().isoformat(), 'cmd': cmd}
        try:
            lock.write_text(json.dumps(payload))
        except Exception as e:
            print(f"Warning: failed to write lock file {lock}: {e}")
            return False
        return True

    def _release_lock(self, model_file):
        lock = self._lock_path(model_file)
        try:
            if lock.exists():
                lock.unlink()
        except Exception as e:
            print(f"Warning: could not remove lock {lock}: {e}")

    def run_inference(self, model_file, threads, n_tokens, extra_args="", timeout=300, stream_output=True):
        """Run a single inference and capture metrics. Streams output live to console and a per-run log file."""
        if self.executable is None:
            print(f"Error: llama-cli executable not found. Skipping run for {model_file}.")
            return None

        model_path = self.models_path / model_file
        if not model_path.exists():
            print(f"Skipping {model_file} - file not found at {model_path}")
            return None

        cmd = (
            f'"{self.executable}" '
            f'-m "{model_path}" '
            f'-t {threads} '
            f'--gpu-layers 0 '
            f'-p "{self.test_prompt}" '
            f'-n {n_tokens} '
            f'{extra_args}'
        )

        # Acquire lock to avoid duplicate runs
        if not self._acquire_lock(model_file, cmd):
            return None

        logs_dir = self.output_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        safe_model = model_file.replace('/', '_').replace('\\', '_')
        log_file = logs_dir / f'{safe_model}_{datetime.now().strftime("%Y%m%dT%H%M%S")}.log'

        print(f"Running: {model_file} with {threads} threads, {n_tokens} tokens (log: {log_file})")

        proc = None
        output_buf = []
        peak_rss = 0
        psutil_available = False
        try:
            import psutil
            psutil_available = True
        except Exception:
            psutil_available = False

        try:
            if stream_output:
                with open(log_file, 'w', encoding='utf-8') as lf:
                    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    start = time.time()
                    last_heartbeat = start
                    while True:
                        # Read any available line
                        line = proc.stdout.readline()
                        if line:
                            lf.write(line)
                            lf.flush()
                            output_buf.append(line)
                            print(line, end='')
                        else:
                            if proc.poll() is not None:
                                break
                            time.sleep(0.1)

                        # heartbeat every 30s
                        now = time.time()
                        if now - last_heartbeat >= 30:
                            elapsed = int(now - start)
                            print(f"... still running ({elapsed}s elapsed) ...")
                            last_heartbeat = now

                        # Sample memory usage if psutil is present
                        if psutil_available and proc and proc.pid:
                            try:
                                p = psutil.Process(proc.pid)
                                rss = p.memory_info().rss
                                if rss > peak_rss:
                                    peak_rss = rss
                            except Exception:
                                pass

                        # timeout
                        now = time.time()
                        if now - start > timeout:
                            print(f"Timeout ({timeout}s) reached for {model_file}; killing process")
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            break

                # read any remaining output from the log file
                with open(log_file, 'r', encoding='utf-8') as lf:
                    output = lf.read()
                returncode = proc.returncode if proc is not None else None
            else:
                # Use Popen to allow sampling of memory even when not streaming
                proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                start = time.time()
                while True:
                    try:
                        out, _ = proc.communicate(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        out = None
                    if out:
                        output_buf.append(out)
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)

                    # sample memory
                    if psutil_available and proc and proc.pid:
                        try:
                            p = psutil.Process(proc.pid)
                            rss = p.memory_info().rss
                            if rss > peak_rss:
                                peak_rss = rss
                        except Exception:
                            pass

                    if time.time() - start > timeout:
                        print(f"Timeout ({timeout}s) reached for {model_file}; killing process")
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        break

                # collect final output
                try:
                    rest, _ = proc.communicate(timeout=1)
                    if rest:
                        output_buf.append(rest)
                except Exception:
                    pass

                output = "".join(output_buf)
                returncode = proc.returncode if proc is not None else None

            # compute peak memory in MB if available
            if peak_rss > 0:
                metrics_mem_mb = int(peak_rss / (1024 * 1024))
            else:
                metrics_mem_mb = 0

            metrics = self.parse_metrics(output)
            metrics['model'] = model_file
            metrics['threads'] = threads
            metrics['n_tokens'] = n_tokens
            metrics['timestamp'] = datetime.now().isoformat()
            metrics['memory_mb'] = metrics_mem_mb

            if metrics.get('tokens_per_sec', 0.0) == 0.0 and metrics.get('total_time_ms', 0.0) == 0.0:
                print("⚠️ Parsed metrics are all zeros — saved full output to log for inspection:")
                print(f"  {log_file}")
                if returncode and returncode != 0:
                    print("Return code:", returncode)

                debug_dir = self.output_dir / 'debug'
                debug_dir.mkdir(exist_ok=True)
                safe_model = model_file.replace('/', '_').replace('\\', '_')
                debug_file = debug_dir / f'debug_{safe_model}_{datetime.now().strftime("%Y%m%dT%H%M%S")}.log'
                with open(debug_file, 'w', encoding='utf-8') as df:
                    df.write(f"Command: {cmd}\n\n")
                    df.write(output)
                print(f"Wrote debug dump to {debug_file}")

            return metrics

        except subprocess.TimeoutExpired:
            print(f"Timeout for {model_file}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            try:
                self._release_lock(model_file)
            except Exception:
                pass
    
    def parse_metrics(self, output):
        """Extract performance metrics from llama.cpp output using regex
        This is more robust: prefer tokens/sec reported on the "eval time" line,
        fall back to any tokens/sec value found in the output. Also tolerate
        commas in numbers and pick the most relevant match.
        """
        metrics = {
            'tokens_per_sec': 0.0,
            'prompt_tokens': 0,
            'generated_tokens': 0,
            'total_time_ms': 0.0,
            'prompt_time_ms': 0.0,
            'generation_time_ms': 0.0,
            'memory_mb': 0
        }

        tokens_per_sec = None

        for line in output.splitlines():
            line_lower = line.lower()

            # Prefer tokens/sec reported on the "eval time" line (more representative of generation)
            if 'eval time' in line_lower:
                m = re.search(r'([\d,\.]+)\s*tokens per second', line_lower)
                if m:
                    tokens_per_sec = float(m.group(1).replace(',', ''))

            # Capture tokens/sec anywhere as a fallback
            if tokens_per_sec is None:
                m_any = re.search(r'([\d,\.]+)\s*tokens per second', line_lower)
                if m_any:
                    tokens_per_sec = float(m_any.group(1).replace(',', ''))

            # Match generation time (eval time)
            match_eval = re.search(r'(?<!prompt\s)eval time\s*=\s*([\d\.]+)\s*ms', line_lower)
            if match_eval:
                metrics['generation_time_ms'] = float(match_eval.group(1))

            # Match prompt eval time
            match_prompt = re.search(r'prompt eval time\s*=\s*([\d\.]+)\s*ms', line_lower)
            if match_prompt:
                metrics['prompt_time_ms'] = float(match_prompt.group(1))

            # Optionally parse total time
            match_total = re.search(r'total time\s*=\s*([\d\.]+)\s*ms', line_lower)
            if match_total:
                metrics['total_time_ms'] = float(match_total.group(1))

        if tokens_per_sec is not None:
            metrics['tokens_per_sec'] = tokens_per_sec

        return metrics

    
    def experiment_1_quantization_comparison(self):
        """Compare different quantization formats"""
        print("\n" + "="*60)
        print("EXPERIMENT 1: Quantization Format Comparison")
        print("="*60)
        
        quant_formats = [
            'phi-2.Q4_K_M.gguf',
            'phi-2.Q5_K_M.gguf',
            'phi-2.Q8_0.gguf',
        ]
        
        results = []
        threads = 4
        n_tokens = 64
        
        for model in quant_formats:
            model_path = self.models_path / model
            if not model_path.exists():
                print(f"Skipping {model} - file not found")
                continue
            
            metrics = self.run_inference(model, threads, n_tokens)
            if metrics:
                results.append(metrics)
                time.sleep(2)
        
        self.save_results(results, 'quantization_comparison.csv')
        return results
    
    def experiment_2_context_length_scaling(self):
        """Test performance with varying context lengths"""
        print("\n" + "="*60)
        print("EXPERIMENT 2: Context Length Scaling")
        print("="*60)
        
        model = 'phi-2.Q4_K_M.gguf'
        threads = 4
        token_counts = [32, 64, 128, 256, 512, 1024]
        
        results = []
        
        for n_tokens in token_counts:
            metrics = self.run_inference(model, threads, n_tokens)
            if metrics:
                results.append(metrics)
                time.sleep(2)
        
        self.save_results(results, 'context_length_scaling.csv')
        return results
    
    def experiment_3_thread_scaling_comprehensive(self):
        """Extended thread scaling with more data points"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: Comprehensive Thread Scaling")
        print("="*60)
        
        model = 'phi-2.Q4_K_M.gguf'
        thread_counts = [1, 2, 3, 4, 6, 8, 12, 16]
        n_tokens = 64
        
        results = []
        
        for threads in thread_counts:
            for run in range(3):
                metrics = self.run_inference(model, threads, n_tokens)
                if metrics:
                    metrics['run'] = run + 1
                    results.append(metrics)
                    time.sleep(1)
        
        self.save_results(results, 'thread_scaling_detailed.csv')
        return results
    
    def experiment_4_memory_options(self):
        """Test memory mapping vs full loading"""
        print("\n" + "="*60)
        print("EXPERIMENT 4: Memory Configuration")
        print("="*60)
        
        model = 'phi-2.Q4_K_M.gguf'
        threads = 4
        n_tokens = 64
        
        results = []
        
        print("Testing with memory mapping...")
        metrics_mmap = self.run_inference(model, threads, n_tokens)
        if metrics_mmap:
            metrics_mmap['config'] = 'mmap_enabled'
            results.append(metrics_mmap)
        
        time.sleep(2)
        
        print("Testing without memory mapping...")
        metrics_no_mmap = self.run_inference(model, threads, n_tokens, '--no-mmap')
        if metrics_no_mmap:
            metrics_no_mmap['config'] = 'mmap_disabled'
            results.append(metrics_no_mmap)
        
        self.save_results(results, 'memory_configuration.csv')
        return results
    
    def experiment_5_model_size_comparison(self):
        """Compare different model sizes"""
        print("\n" + "="*60)
        print("EXPERIMENT 5: Model Size Comparison")
        print("="*60)
        
        # Note: Model size comparison with single model (tinyllama excluded)
        models = [
            ('phi-2.Q4_K_M.gguf', '2.7B'),
        ]
        
        threads = 4
        n_tokens = 64
        
        results = []
        
        for model_file, size in models:
            model_path = self.models_path / model_file
            if not model_path.exists():
                print(f"Skipping {model_file} - file not found")
                continue
            
            metrics = self.run_inference(model_file, threads, n_tokens)
            if metrics:
                metrics['model_size'] = size
                results.append(metrics)
                time.sleep(2)
        
        self.save_results(results, 'model_size_comparison.csv')
        return results
    
    def save_results(self, results, filename):
        """Save results to CSV"""
        if not results:
            print(f"No results to save for {filename}")
            return
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ Results saved to {output_path}")
    
    def run_all_experiments(self):
        """Run complete benchmark suite"""
        print("\n" + "="*60)
        print("STARTING COMPREHENSIVE BENCHMARK SUITE")
        print("="*60)
        
        all_results = {}
        
        all_results['quantization'] = self.experiment_1_quantization_comparison()
        all_results['context_length'] = self.experiment_2_context_length_scaling()
        all_results['thread_scaling'] = self.experiment_3_thread_scaling_comprehensive()
        all_results['memory_config'] = self.experiment_4_memory_options()
        all_results['model_size'] = self.experiment_5_model_size_comparison()
        
        self.generate_summary_report(all_results)
        
        print("\n" + "="*60)
        print("BENCHMARK SUITE COMPLETE!")
        print("="*60)
        print(f"Results saved to: {self.output_dir}")
    
    def generate_summary_report(self, all_results):
        """Generate a summary markdown report"""
        report_path = self.output_dir / 'summary_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# LLM Inference Optimization - Experimental Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for exp_name, results in all_results.items():
                if results:
                    f.write(f"## {exp_name.replace('_', ' ').title()}\n\n")
                    f.write(f"Total runs: {len(results)}\n\n")
                    
                    avg_tps = sum(r['tokens_per_sec'] for r in results) / len(results)
                    f.write(f"Average tokens/sec: {avg_tps:.2f}\n\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("- Analyze CSV files for detailed patterns\n")
            f.write("- Create visualizations\n")
            f.write("- Update report with findings\n")
        
        print(f"✓ Summary report saved to {report_path}")


# ============================================================================

if __name__ == "__main__":
    # Configure paths for your system
    LLAMA_CPP_PATH = r"C:\Users\steve\OneDrive\Documentos\MasterFirstSemester\hpc\llm-inference-optimization\llama.cpp"
    MODELS_PATH = r"C:\Users\steve\OneDrive\Documentos\MasterFirstSemester\hpc\llm-inference-optimization\models"
    
    # Create benchmark suite
    suite = LLMBenchmarkSuite(LLAMA_CPP_PATH, MODELS_PATH)
    
    # Run all experiments
    suite.run_all_experiments()
