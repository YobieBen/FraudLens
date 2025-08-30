#!/usr/bin/env python3
"""
FraudLens Comprehensive Test Runner
Runs all tests and generates reports
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse


class TestRunner:
    """Comprehensive test runner with reporting"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_command(self, command: str, description: str) -> Dict[str, Any]:
        """Run a command and capture output"""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"{'='*60}")
        
        if self.verbose:
            print(f"Command: {command}")
        
        start = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ {description} - PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {description} - FAILED ({duration:.2f}s)")
                if self.verbose:
                    print(f"Error: {result.stderr}")
            
            return {
                'command': command,
                'description': description,
                'success': success,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            print(f"⏱️ {description} - TIMEOUT")
            return {
                'command': command,
                'description': description,
                'success': False,
                'duration': 300,
                'error': 'Timeout exceeded'
            }
        except Exception as e:
            print(f"❌ {description} - ERROR: {e}")
            return {
                'command': command,
                'description': description,
                'success': False,
                'duration': 0,
                'error': str(e)
            }
    
    def run_unit_tests(self):
        """Run unit tests"""
        return self.run_command(
            "pytest tests/unit/ -v --cov=fraudlens --cov-report=term-missing",
            "Unit Tests"
        )
    
    def run_integration_tests(self):
        """Run integration tests"""
        return self.run_command(
            "pytest tests/integration/ -v -m integration",
            "Integration Tests"
        )
    
    def run_performance_tests(self):
        """Run performance benchmarks"""
        return self.run_command(
            "pytest tests/performance/ -v -m benchmark --benchmark-only --benchmark-json=benchmark.json",
            "Performance Benchmarks"
        )
    
    def run_email_tests(self):
        """Run email-specific tests"""
        return self.run_command(
            "pytest tests/unit/test_gmail_integration.py tests/integration/test_email_api_endpoints.py -v",
            "Email Tests"
        )
    
    def run_security_checks(self):
        """Run security analysis"""
        return self.run_command(
            "bandit -r fraudlens/ -f json -o security-report.json",
            "Security Analysis"
        )
    
    def run_code_quality(self):
        """Run code quality checks"""
        commands = [
            ("black --check fraudlens/ tests/", "Code Formatting Check"),
            ("flake8 fraudlens/ tests/ --max-line-length=100", "Flake8 Linting"),
            ("mypy fraudlens/ --ignore-missing-imports", "Type Checking")
        ]
        
        results = []
        for cmd, desc in commands:
            results.append(self.run_command(cmd, desc))
        
        return results
    
    def generate_coverage_report(self):
        """Generate coverage report"""
        return self.run_command(
            "coverage html && coverage report --fail-under=80",
            "Coverage Report Generation"
        )
    
    def run_all_tests(self):
        """Run all test suites"""
        self.start_time = datetime.now()
        
        print("\n" + "="*60)
        print("FRAUDLENS COMPREHENSIVE TEST SUITE")
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Run tests in order
        self.results['unit'] = self.run_unit_tests()
        self.results['integration'] = self.run_integration_tests()
        self.results['email'] = self.run_email_tests()
        self.results['performance'] = self.run_performance_tests()
        self.results['security'] = self.run_security_checks()
        self.results['quality'] = self.run_code_quality()
        self.results['coverage'] = self.generate_coverage_report()
        
        self.end_time = datetime.now()
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary report"""
        print("\n" + "="*60)
        print("TEST SUMMARY REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = []
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Process results
        for category, result in self.results.items():
            if isinstance(result, list):
                # Multiple results (code quality)
                for r in result:
                    total_tests += 1
                    if r['success']:
                        passed_tests += 1
                    else:
                        failed_tests.append(r['description'])
            else:
                # Single result
                total_tests += 1
                if result['success']:
                    passed_tests += 1
                else:
                    failed_tests.append(result['description'])
        
        # Print summary
        print(f"\nTotal Test Categories: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"  - {test}")
        
        # Coverage information
        try:
            with open('coverage.json', 'r') as f:
                coverage_data = json.load(f)
                total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                print(f"\n📊 Code Coverage: {total_coverage:.1f}%")
        except:
            pass
        
        # Performance benchmarks
        try:
            with open('benchmark.json', 'r') as f:
                benchmark_data = json.load(f)
                benchmarks = benchmark_data.get('benchmarks', [])
                if benchmarks:
                    print("\n⚡ Performance Highlights:")
                    for bench in benchmarks[:3]:
                        name = bench.get('name', 'Unknown')
                        mean = bench.get('stats', {}).get('mean', 0) * 1000
                        print(f"  - {name}: {mean:.2f}ms average")
        except:
            pass
        
        # Security findings
        try:
            with open('security-report.json', 'r') as f:
                security_data = json.load(f)
                issues = security_data.get('results', [])
                print(f"\n🔒 Security Issues Found: {len(issues)}")
                if issues:
                    severity_counts = {}
                    for issue in issues:
                        severity = issue.get('issue_severity', 'UNKNOWN')
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    for severity, count in severity_counts.items():
                        print(f"  - {severity}: {count}")
        except:
            pass
        
        # Final status
        print("\n" + "="*60)
        if passed_tests == total_tests:
            print("✅ ALL TESTS PASSED! 🎉")
            sys.exit(0)
        else:
            print(f"❌ {len(failed_tests)} TEST(S) FAILED")
            sys.exit(1)
    
    def run_specific_test(self, test_type: str):
        """Run specific test type"""
        test_map = {
            'unit': self.run_unit_tests,
            'integration': self.run_integration_tests,
            'performance': self.run_performance_tests,
            'email': self.run_email_tests,
            'security': self.run_security_checks,
            'quality': self.run_code_quality,
            'coverage': self.generate_coverage_report
        }
        
        if test_type in test_map:
            result = test_map[test_type]()
            if isinstance(result, list):
                success = all(r['success'] for r in result)
            else:
                success = result['success']
            
            if success:
                print(f"\n✅ {test_type.upper()} tests passed!")
                sys.exit(0)
            else:
                print(f"\n❌ {test_type.upper()} tests failed!")
                sys.exit(1)
        else:
            print(f"Unknown test type: {test_type}")
            print(f"Available types: {', '.join(test_map.keys())}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FraudLens Test Runner')
    parser.add_argument(
        '--type',
        choices=['all', 'unit', 'integration', 'performance', 'email', 'security', 'quality', 'coverage'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first failure'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    required_packages = ['pytest', 'coverage', 'pytest-cov', 'pytest-benchmark']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)
    
    # Run tests
    runner = TestRunner(verbose=args.verbose)
    
    if args.type == 'all':
        runner.run_all_tests()
    else:
        runner.run_specific_test(args.type)


if __name__ == "__main__":
    main()