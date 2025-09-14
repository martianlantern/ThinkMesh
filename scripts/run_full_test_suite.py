#!/usr/bin/env python3
"""
Comprehensive test runner for ThinkMesh.
"""
import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any


class TestRunner:
    """Comprehensive test runner for ThinkMesh."""
    
    def __init__(self, verbose: bool = True, gpu_tests: bool = False, a100_tests: bool = False):
        self.verbose = verbose
        self.gpu_tests = gpu_tests
        self.a100_tests = a100_tests
        self.results: Dict[str, Any] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_command(self, cmd: List[str], timeout: int = 300) -> Dict[str, Any]:
        """Run a command and capture results."""
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent
            )
            
            execution_time = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "execution_time": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": time.time() - start_time
            }
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        print("\n" + "="*60)
        print("RUNNING UNIT TESTS")
        print("="*60)
        
        # Basic unit tests (no GPU required)
        basic_cmd = [
            "pytest", "tests/unit/", 
            "-v", "--tb=short", 
            "-m", "unit and not gpu",
            "--timeout=60"
        ]
        
        result = self.run_command(basic_cmd, timeout=300)
        self.results["unit_tests_basic"] = result
        
        if not result["success"]:
            print("❌ Basic unit tests failed")
            print(result["stderr"])
            return False
        
        print("✅ Basic unit tests passed")
        
        # GPU unit tests if requested
        if self.gpu_tests:
            gpu_cmd = [
                "pytest", "tests/unit/",
                "-v", "--tb=short",
                "-m", "unit and gpu and not a100",
                "--timeout=120"
            ]
            
            gpu_result = self.run_command(gpu_cmd, timeout=600)
            self.results["unit_tests_gpu"] = gpu_result
            
            if not gpu_result["success"]:
                print("⚠️  GPU unit tests failed (continuing with other tests)")
                print(gpu_result["stderr"])
            else:
                print("✅ GPU unit tests passed")
        
        return True
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("\n" + "="*60)
        print("RUNNING INTEGRATION TESTS")
        print("="*60)
        
        # Basic integration tests
        basic_cmd = [
            "pytest", "tests/integration/",
            "-v", "--tb=short",
            "-m", "integration and not gpu and not slow",
            "--timeout=120"
        ]
        
        result = self.run_command(basic_cmd, timeout=600)
        self.results["integration_tests_basic"] = result
        
        if not result["success"]:
            print("❌ Basic integration tests failed")
            print(result["stderr"])
            return False
        
        print("✅ Basic integration tests passed")
        
        # GPU integration tests if requested
        if self.gpu_tests:
            gpu_cmd = [
                "pytest", "tests/integration/",
                "-v", "--tb=short", 
                "-m", "integration and gpu and not slow and not a100",
                "--timeout=300"
            ]
            
            gpu_result = self.run_command(gpu_cmd, timeout=900)
            self.results["integration_tests_gpu"] = gpu_result
            
            if not gpu_result["success"]:
                print("⚠️  GPU integration tests failed (continuing with other tests)")
                print(gpu_result["stderr"])
            else:
                print("✅ GPU integration tests passed")
        
        return True
    
    def run_benchmark_tests(self) -> bool:
        """Run benchmark tests."""
        print("\n" + "="*60)
        print("RUNNING BENCHMARK TESTS")
        print("="*60)
        
        # GSM8K utils tests
        utils_cmd = [
            "pytest", "tests/benchmarks/test_gsm8k.py::TestGSM8KUtils",
            "-v", "--tb=short",
            "--timeout=60"
        ]
        
        result = self.run_command(utils_cmd, timeout=180)
        self.results["benchmark_utils"] = result
        
        if not result["success"]:
            print("❌ Benchmark utils tests failed")
            print(result["stderr"])
            return False
        
        print("✅ Benchmark utils tests passed")
        
        # Quick benchmark test (CPU only)
        quick_cmd = [
            "python", "scripts/run_benchmarks.py",
            "--model", "small_cpu",
            "--strategies", "self_consistency_small",
            "--quick"
        ]
        
        quick_result = self.run_command(quick_cmd, timeout=300)
        self.results["benchmark_quick"] = quick_result
        
        if not quick_result["success"]:
            print("⚠️  Quick benchmark test failed (continuing with other tests)")
            print(quick_result["stderr"])
        else:
            print("✅ Quick benchmark test passed")
        
        return True
    
    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        if not (self.gpu_tests or self.a100_tests):
            print("\n⏭️  Skipping performance tests (requires --gpu or --a100)")
            return True
        
        print("\n" + "="*60)
        print("RUNNING PERFORMANCE TESTS") 
        print("="*60)
        
        if self.a100_tests:
            # A100-specific performance tests
            a100_cmd = [
                "pytest", "tests/performance/",
                "-v", "--tb=short",
                "-m", "performance and a100",
                "--timeout=1800"
            ]
            
            result = self.run_command(a100_cmd, timeout=2400)
            self.results["performance_a100"] = result
            
            if not result["success"]:
                print("⚠️  A100 performance tests failed")
                print(result["stderr"])
                return False
            
            print("✅ A100 performance tests passed")
            
        elif self.gpu_tests:
            # Regular GPU performance tests
            gpu_cmd = [
                "pytest", "tests/performance/",
                "-v", "--tb=short", 
                "-m", "performance and gpu and not a100",
                "--timeout=900"
            ]
            
            result = self.run_command(gpu_cmd, timeout=1200)
            self.results["performance_gpu"] = result
            
            if not result["success"]:
                print("⚠️  GPU performance tests failed")
                print(result["stderr"])
                return False
            
            print("✅ GPU performance tests passed")
        
        return True
    
    def run_linting_and_formatting(self) -> bool:
        """Run code quality checks."""
        print("\n" + "="*60)
        print("RUNNING CODE QUALITY CHECKS")
        print("="*60)
        
        # Ruff linting
        lint_cmd = ["ruff", "check", "src/", "tests/", "--output-format=text"]
        lint_result = self.run_command(lint_cmd, timeout=120)
        self.results["linting"] = lint_result
        
        if not lint_result["success"]:
            print("❌ Linting failed")
            print(lint_result["stdout"])
            print(lint_result["stderr"])
            return False
        
        print("✅ Linting passed")
        
        # Ruff formatting check
        format_cmd = ["ruff", "format", "--check", "src/", "tests/"]
        format_result = self.run_command(format_cmd, timeout=60)
        self.results["formatting"] = format_result
        
        if not format_result["success"]:
            print("❌ Formatting check failed")
            print(format_result["stdout"])
            return False
        
        print("✅ Formatting check passed")
        
        return True
    
    def run_type_checking(self) -> bool:
        """Run type checking with mypy."""
        print("\n" + "="*60)
        print("RUNNING TYPE CHECKING")
        print("="*60)
        
        mypy_cmd = ["mypy", "src/thinkmesh/", "--ignore-missing-imports"]
        result = self.run_command(mypy_cmd, timeout=120)
        self.results["type_checking"] = result
        
        if not result["success"]:
            print("⚠️  Type checking found issues (not blocking)")
            print(result["stdout"])
            return True  # Non-blocking for now
        
        print("✅ Type checking passed")
        return True
    
    def check_dependencies(self) -> bool:
        """Check that required dependencies are installed."""
        print("\n" + "="*60)
        print("CHECKING DEPENDENCIES")
        print("="*60)
        
        required_packages = [
            "torch",
            "transformers", 
            "pytest",
            "ruff",
            "mypy",
            "pydantic",
            "typer"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} - MISSING")
                missing_packages.append(package)
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✅ CUDA available: {gpu_count} GPU(s), {gpu_memory:.1f}GB memory")
                
                if gpu_memory > 70:
                    print("✅ A100-class GPU detected")
                    
            else:
                print("⚠️  CUDA not available")
        except Exception as e:
            print(f"❌ GPU check failed: {e}")
        
        if missing_packages:
            print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        return True
    
    def generate_report(self):
        """Generate test execution report."""
        print("\n" + "="*60)
        print("TEST EXECUTION REPORT")
        print("="*60)
        
        total_time = sum(
            result.get("execution_time", 0) 
            for result in self.results.values()
        )
        
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Tests run: {len(self.results)}")
        
        successful_tests = sum(
            1 for result in self.results.values() 
            if result.get("success", False)
        )
        
        print(f"Successful: {successful_tests}")
        print(f"Failed: {len(self.results) - successful_tests}")
        
        print("\nDetailed Results:")
        print("-" * 40)
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            time_str = f"{result.get('execution_time', 0):.1f}s"
            print(f"{test_name:25} {status} ({time_str})")
            
            if not result.get("success", False) and result.get("stderr"):
                print(f"  Error: {result['stderr'][:100]}...")
        
        # Save detailed report
        report_path = Path("test_results") / "test_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"ThinkMesh Test Report\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total time: {total_time:.1f}s\n")
            f.write(f"Success rate: {successful_tests}/{len(self.results)}\n\n")
            
            for test_name, result in self.results.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"Test: {test_name}\n")
                f.write(f"Success: {result.get('success', False)}\n")
                f.write(f"Time: {result.get('execution_time', 0):.1f}s\n")
                f.write(f"Return code: {result.get('returncode', 'N/A')}\n")
                
                if result.get('stdout'):
                    f.write(f"\nSTDOUT:\n{result['stdout']}\n")
                
                if result.get('stderr'):
                    f.write(f"\nSTDERR:\n{result['stderr']}\n")
        
        print(f"\nDetailed report saved to: {report_path}")
        
        return successful_tests == len(self.results)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run ThinkMesh test suite")
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests")
    parser.add_argument("--a100", action="store_true", help="Run A100-specific tests")
    parser.add_argument("--quick", action="store_true", help="Run only essential tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    
    args = parser.parse_args()
    
    # A100 implies GPU
    if args.a100:
        args.gpu = True
    
    runner = TestRunner(verbose=args.verbose, gpu_tests=args.gpu, a100_tests=args.a100)
    
    print("ThinkMesh Comprehensive Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not args.skip_deps and not runner.check_dependencies():
        print("\n❌ Dependency check failed. Fix dependencies and try again.")
        return 1
    
    success = True
    
    try:
        # Core functionality tests
        if not runner.run_linting_and_formatting():
            success = False
            if not args.quick:  # In quick mode, continue despite linting failures
                return 1
        
        if not runner.run_unit_tests():
            success = False
            return 1
        
        if not runner.run_integration_tests():
            success = False
            return 1
        
        # Skip slower tests in quick mode
        if not args.quick:
            if not runner.run_benchmark_tests():
                success = False
            
            if not runner.run_performance_tests():
                success = False
            
            runner.run_type_checking()  # Non-blocking
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        success = False
    
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        success = False
    
    finally:
        # Always generate report
        report_success = runner.generate_report()
        success = success and report_success
    
    if success:
        print("\n🎉 All tests passed! ThinkMesh is ready for production.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the report for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
