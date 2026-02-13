#!/usr/bin/env python3
"""
Benchmark Data Collection Script
Runs comprehensive benchmarks and collects performance metrics
"""

import subprocess
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List
import argparse

class BenchmarkCollector:
    """Collects and analyzes benchmark data from C++ executables"""
    
    def __init__(self, build_dir: str = "cpp/build/Release"):
        self.build_dir = Path(build_dir)
        self.results = {}
        
    def run_benchmark(self, executable: str, args: List[str] = None, runs: int = 10) -> Dict:
        """Run a benchmark executable multiple times and collect stats"""
        
        exe_path = self.build_dir / f"{executable}.exe"
        
        if not exe_path.exists():
            print(f"⚠ {executable} not found at {exe_path}")
            return None
        
        print(f"\nRunning {executable} ({runs} iterations)...")
        
        latencies = []
        
        for i in range(runs):
            start = time.time()
            
            try:
                cmd = [str(exe_path)] + (args or [])
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                end = time.time()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                
                print(f"  Run {i+1}/{runs}: {latency_ms:.1f}ms", end='\r')
                
            except subprocess.TimeoutExpired:
                print(f"  Run {i+1}/{runs}: TIMEOUT")
                continue
            except Exception as e:
                print(f"  Run {i+1}/{runs}: ERROR - {e}")
                continue
        
        if not latencies:
            return None
        
        # Calculate statistics
        stats = {
            "executable": executable,
            "runs": len(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "p50_ms": statistics.quantiles(latencies, n=100)[49] if len(latencies) >= 2 else latencies[0],
            "p95_ms": statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 2 else latencies[0],
            "p99_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 2 else latencies[0],
        }
        
        print(f"\n  ✓ {executable}: {stats['mean_ms']:.1f}ms (±{stats['stdev_ms']:.1f}ms)")
        
        return stats
    
    def collect_all_benchmarks(self):
        """Run all available benchmarks"""
        
        print("=== Collecting Benchmark Data ===\n")
        
        benchmarks = [
            ("bench_engine", ["--runs", "100"]),
            ("bench_memory", ["--duration", "60"]),
            ("bench_retrieval", ["--samples", "100"]),
            ("ablation_suite", []),
        ]
        
        for exe, args in benchmarks:
            result = self.run_benchmark(exe, args, runs=5)
            if result:
                self.results[exe] = result
        
        return self.results
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save results to JSON file"""
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")
    
    def print_summary(self):
        """Print summary table"""
        
        print("\n=== Benchmark Summary ===\n")
        print(f"{'Benchmark':<25} {'Mean (ms)':<12} {'p95 (ms)':<12} {'Runs':<8}")
        print("-" * 60)
        
        for exe, stats in self.results.items():
            print(f"{exe:<25} {stats['mean_ms']:<12.1f} {stats['p95_ms']:<12.1f} {stats['runs']:<8}")
        
        print()

def main():
    parser = argparse.ArgumentParser(description="Collect benchmark data")
    parser.add_argument("--build-dir", default="cpp/build/Release", help="Build directory")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    parser.add_argument("--runs", type=int, default=10, help="Runs per benchmark")
    
    args = parser.parse_args()
    
    collector = BenchmarkCollector(args.build_dir)
    collector.collect_all_benchmarks()
    collector.save_results(args.output)
    collector.print_summary()

if __name__ == "__main__":
    main()
