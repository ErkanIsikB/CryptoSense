"""Centralized test runner.

Discovers and executes all unit tests in the tests/ directory and runs
the transaction-isolated database integration test. Compiles and saves
a detailed Markdown test report to test_report.md.
"""
import sys
import os
import time
import unittest
import io

# Adjust path to find src and scripts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from tests.test_integration import run_integration_tests
from src.db.db import close_pool

def run_suite():
    # Discover tests in tests/
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(ROOT_DIR, "tests"), pattern="test_*.py")
    
    # We will capture output from the runner
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    
    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time
    
    return result, stream.getvalue(), duration

def main():
    print("=========================================")
    print("🚀 Running CryptoSense Test Suite")
    print("=========================================")
    
    # 1. Run unit tests
    unit_result, unit_log, unit_duration = run_suite()
    
    # 2. Run integration tests
    integration_passed = True
    integration_error = ""
    integration_start = time.time()
    
    integration_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = integration_stream
    
    try:
        run_integration_tests()
    except Exception as e:
        integration_passed = False
        import traceback
        traceback.print_exc()
        integration_error = str(e)
    finally:
        sys.stdout = original_stdout
        close_pool()
        
    integration_duration = time.time() - integration_start
    integration_log = integration_stream.getvalue()
    
    # 3. Run XQuik Live API tests
    xquick_passed = True
    xquick_error = ""
    xquick_start = time.time()
    
    xquick_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = xquick_stream
    
    try:
        import asyncio
        from tests.test_xquick import test_live_tweets
        asyncio.run(test_live_tweets())
    except Exception as e:
        xquick_passed = False
        import traceback
        traceback.print_exc()
        xquick_error = str(e)
    finally:
        sys.stdout = original_stdout
        
    xquick_duration = time.time() - xquick_start
    xquick_log = xquick_stream.getvalue()
    
    # 4. Compile report
    total_tests = unit_result.testsRun
    failed_tests = len(unit_result.failures) + len(unit_result.errors)
    passed_tests = total_tests - failed_tests
    
    status = "SUCCESS" if (failed_tests == 0 and integration_passed and xquick_passed) else "FAILED"
    
    report = []
    report.append("# CryptoSense Automated Test Suite Execution Report")
    report.append("")
    report.append(f"**Execution Status**: {'🟢 PASS' if status == 'SUCCESS' else '🔴 FAIL'}")
    report.append(f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## Executive Summary")
    report.append("| Test Suite | Total Tests | Passed | Failed | Duration |")
    report.append("| :--- | :--- | :--- | :--- | :--- |")
    report.append(f"| Unit Tests | {total_tests} | {passed_tests} | {failed_tests} | {unit_duration:.3f}s |")
    report.append(f"| Database Integration | 2 | {2 if integration_passed else 0} | {0 if integration_passed else 2} | {integration_duration:.3f}s |")
    report.append(f"| XQuik Live API | 1 | {1 if xquick_passed else 0} | {0 if xquick_passed else 1} | {xquick_duration:.3f}s |")
    report.append("")
    
    if failed_tests > 0 or not integration_passed or not xquick_passed:
        report.append("## ❌ Failures Details")
        for failure in unit_result.failures:
            report.append(f"### Unit Test Failure: `{failure[0]}`")
            report.append("```")
            report.append(failure[1])
            report.append("```")
            report.append("")
        for error in unit_result.errors:
            report.append(f"### Unit Test Error: `{error[0]}`")
            report.append("```")
            report.append(error[1])
            report.append("```")
            report.append("")
        if not integration_passed:
            report.append("### Database Integration Test Failure")
            report.append("```")
            report.append(integration_log)
            if integration_error:
                report.append(f"Error details: {integration_error}")
            report.append("```")
            report.append("")
        if not xquick_passed:
            report.append("### XQuik Live API Test Failure")
            report.append("```")
            report.append(xquick_log)
            if xquick_error:
                report.append(f"Error details: {xquick_error}")
            report.append("```")
            report.append("")
            
    report.append("## Detailed Execution Logs")
    report.append("### Unit Tests")
    report.append("```")
    report.append(unit_log)
    report.append("```")
    report.append("")
    report.append("### Database Integration (Transaction-Isolated)")
    report.append("```")
    report.append(integration_log)
    report.append("```")
    report.append("")
    report.append("### XQuik Live API")
    report.append("```")
    report.append(xquick_log)
    report.append("```")
    report.append("")
    
    report.append("---")
    report.append("*Report generated automatically by `scripts/run_all_tests.py`.*")
    
    report_content = "\n".join(report)
    
    # Save report
    report_path = os.path.join(ROOT_DIR, "test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report saved to: {report_path}")
    print(f"Overall Status: {status}")
    print("=========================================")
    
    if status == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
