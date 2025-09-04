"""
End-to-End tests for Gradio application.

Author: Yobie Benjamin
Date: 2025-08-27 19:00:00 PDT
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests
from PIL import Image
from loguru import logger


class GradioE2ETester:
    """Comprehensive E2E tester for Gradio application."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/predict"
        self.results = []
        self.passed = 0
        self.failed = 0

    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        result = {
            "test": test_name,
            "passed": passed,
            "status": status,
            "details": details,
            "timestamp": time.time(),
        }
        self.results.append(result)

        if passed:
            self.passed += 1
            logger.success(f"{status} - {test_name}")
        else:
            self.failed += 1
            logger.error(f"{status} - {test_name}: {details}")

    def create_test_image(self, path: Path):
        """Create a test image for testing."""
        # Create a simple test image
        img = Image.new("RGB", (640, 480), color="white")
        pixels = img.load()

        # Add some patterns to make it look like content
        for i in range(0, 640, 20):
            for j in range(0, 480, 20):
                color = ((i * 255) // 640, (j * 255) // 480, 128)
                for x in range(i, min(i + 10, 640)):
                    for y in range(j, min(j + 10, 480)):
                        pixels[x, y] = color

        # Add text-like elements
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)
        try:
            # Try to use a basic font
            draw.text((50, 50), "TEST DOCUMENT", fill=(0, 0, 0))
            draw.text((50, 100), "ID: 12345678", fill=(0, 0, 0))
            draw.text((50, 150), "Name: John Doe", fill=(0, 0, 0))
            draw.text((50, 200), "Date: 2025-08-27", fill=(0, 0, 0))
        except:
            pass

        img.save(path)
        return path

    def create_test_pdf(self, path: Path):
        """Create a test PDF for testing."""
        # Create a simple PDF with reportlab or just use a text file as fallback
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            c = canvas.Canvas(str(path), pagesize=letter)
            c.drawString(100, 750, "TEST FINANCIAL DOCUMENT")
            c.drawString(100, 700, "Transaction ID: TXN-2025-001")
            c.drawString(100, 650, "Amount: $10,000")
            c.drawString(100, 600, "Date: 2025-08-27")
            c.drawString(100, 550, "Status: Pending Review")
            c.save()
        except ImportError:
            # Fallback: create a text file with .pdf extension
            with open(path, "wb") as f:
                f.write(b"%PDF-1.4\n")
                f.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
                f.write(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
                f.write(
                    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << >> /MediaBox [0 0 612 792] >>\nendobj\n"
                )
                f.write(
                    b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n"
                )
                f.write(b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n231\n%%EOF\n")

        return path

    async def test_server_health(self):
        """Test if server is running and responsive."""
        test_name = "Server Health Check"
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                self.log_result(test_name, True, "Server is running")
                return True
            else:
                self.log_result(test_name, False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_result(test_name, False, str(e))
            return False

    async def test_text_analysis(self):
        """Test text fraud detection."""
        test_name = "Text Analysis"

        test_cases = [
            {
                "text": "Congratulations! You've won $1,000,000! Click here to claim your prize now! This is not a scam, 100% legitimate!",
                "fraud_type": "phishing",
                "expected_high_fraud": True,
            },
            {
                "text": "Your account statement for March 2025 is now available. Please log in to view your transactions.",
                "fraud_type": "auto",
                "expected_high_fraud": False,
            },
            {
                "text": "URGENT: Your account will be suspended unless you verify your identity immediately. Click this link: bit.ly/verify-now",
                "fraud_type": "phishing",
                "expected_high_fraud": True,
            },
        ]

        all_passed = True
        for i, test_case in enumerate(test_cases):
            try:
                # Make API call to Gradio endpoint
                data = {
                    "fn_index": 0,  # Text analysis function
                    "data": [test_case["text"], test_case["fraud_type"]],
                }

                response = requests.post(self.api_url, json=data, timeout=10)
                result = response.json()

                if "data" in result:
                    # Parse the result
                    fraud_score_text = result["data"][0] if len(result["data"]) > 0 else ""

                    # Check if fraud was detected as expected
                    if test_case["expected_high_fraud"]:
                        if (
                            "high" in fraud_score_text.lower()
                            or "fraud" in fraud_score_text.lower()
                        ):
                            logger.info(f"Test case {i+1} passed: High fraud detected as expected")
                        else:
                            logger.warning(
                                f"Test case {i+1}: Expected high fraud but got: {fraud_score_text}"
                            )
                            all_passed = False
                    else:
                        if (
                            "low" in fraud_score_text.lower()
                            or "legitimate" in fraud_score_text.lower()
                        ):
                            logger.info(f"Test case {i+1} passed: Low fraud detected as expected")
                        else:
                            logger.warning(
                                f"Test case {i+1}: Expected low fraud but got: {fraud_score_text}"
                            )
                else:
                    logger.error(f"Test case {i+1} failed: Invalid response format")
                    all_passed = False

            except Exception as e:
                logger.error(f"Test case {i+1} failed: {e}")
                all_passed = False

        self.log_result(
            test_name,
            all_passed,
            "All text analysis tests passed" if all_passed else "Some tests failed",
        )
        return all_passed

    async def test_document_analysis(self):
        """Test document fraud detection."""
        test_name = "Document Analysis"

        try:
            # Create test documents
            test_dir = Path("/tmp/fraudlens_e2e_tests")
            test_dir.mkdir(exist_ok=True)

            # Test passport image
            passport_path = test_dir / "test_passport.jpg"
            self.create_test_image(passport_path)

            # Test PDF document
            pdf_path = test_dir / "test_document.pdf"
            self.create_test_pdf(pdf_path)

            test_files = [
                {"path": passport_path, "doc_type": "passport", "name": "Passport Analysis"},
                {"path": pdf_path, "doc_type": "invoice", "name": "PDF Analysis"},
            ]

            all_passed = True
            for test_file in test_files:
                try:
                    # Read file and encode
                    with open(test_file["path"], "rb") as f:
                        file_data = base64.b64encode(f.read()).decode()

                    # Make API call
                    data = {
                        "fn_index": 1,  # Document analysis function
                        "data": [
                            {
                                "name": test_file["path"].name,
                                "data": f"data:application/octet-stream;base64,{file_data}",
                            },
                            test_file["doc_type"],
                        ],
                    }

                    response = requests.post(self.api_url, json=data, timeout=15)
                    result = response.json()

                    if "data" in result:
                        logger.info(f"{test_file['name']} completed successfully")
                    else:
                        logger.warning(f"{test_file['name']} returned unexpected format")
                        all_passed = False

                except Exception as e:
                    logger.error(f"{test_file['name']} failed: {e}")
                    all_passed = False

            self.log_result(test_name, all_passed, "Document analysis tests completed")
            return all_passed

        except Exception as e:
            self.log_result(test_name, False, str(e))
            return False

    async def test_image_analysis(self):
        """Test image fraud detection."""
        test_name = "Image Analysis"

        try:
            # Create test images
            test_dir = Path("/tmp/fraudlens_e2e_tests")
            test_dir.mkdir(exist_ok=True)

            # Create different types of test images
            test_images = []
            for i in range(3):
                img_path = test_dir / f"test_image_{i}.jpg"
                self.create_test_image(img_path)
                test_images.append(img_path)

            all_passed = True
            for img_path in test_images:
                try:
                    # Read and encode image
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()

                    # Make API call
                    data = {
                        "fn_index": 2,  # Image analysis function
                        "data": [
                            {"name": img_path.name, "data": f"data:image/jpeg;base64,{img_data}"}
                        ],
                    }

                    response = requests.post(self.api_url, json=data, timeout=15)
                    result = response.json()

                    if "data" in result:
                        logger.info(f"Image {img_path.name} analyzed successfully")
                    else:
                        logger.warning(f"Image {img_path.name} returned unexpected format")
                        all_passed = False

                except Exception as e:
                    logger.error(f"Image {img_path.name} analysis failed: {e}")
                    all_passed = False

            self.log_result(test_name, all_passed, "Image analysis tests completed")
            return all_passed

        except Exception as e:
            self.log_result(test_name, False, str(e))
            return False

    async def test_transaction_monitoring(self):
        """Test transaction monitoring."""
        test_name = "Transaction Monitoring"

        try:
            # Generate test transactions
            transactions = []
            for i in range(10):
                amount = np.random.uniform(10, 10000)
                if i % 3 == 0:
                    # Make some transactions suspicious
                    amount = np.random.uniform(50000, 100000)

                transactions.append(
                    {
                        "id": f"TXN-{i:04d}",
                        "amount": round(amount, 2),
                        "currency": "USD",
                        "timestamp": f"2025-08-27T{10+i}:00:00Z",
                        "sender": f"user_{i}",
                        "receiver": f"merchant_{i % 3}",
                    }
                )

            # Convert to JSON string
            transactions_json = json.dumps(transactions, indent=2)

            # Make API call
            data = {
                "fn_index": 3,  # Transaction monitoring function
                "data": [transactions_json, "real-time", 7000],
            }

            response = requests.post(self.api_url, json=data, timeout=10)
            result = response.json()

            if "data" in result:
                self.log_result(test_name, True, "Transaction monitoring completed")
                return True
            else:
                self.log_result(test_name, False, "Unexpected response format")
                return False

        except Exception as e:
            self.log_result(test_name, False, str(e))
            return False

    async def test_compliance_checks(self):
        """Test compliance features."""
        test_name = "Compliance Checks"

        try:
            # Test GDPR compliance check
            compliance_data = {
                "user_id": "user_12345",
                "data_type": "transaction_history",
                "purpose": "fraud_detection",
                "retention_days": 90,
            }

            # Make API call
            data = {
                "fn_index": 4,  # Compliance function
                "data": [json.dumps(compliance_data, indent=2), "GDPR"],
            }

            response = requests.post(self.api_url, json=data, timeout=10)
            result = response.json()

            if "data" in result:
                self.log_result(test_name, True, "Compliance checks completed")
                return True
            else:
                self.log_result(test_name, False, "Unexpected response format")
                return False

        except Exception as e:
            self.log_result(test_name, False, str(e))
            return False

    async def run_all_tests(self):
        """Run all E2E tests."""
        logger.info("=" * 80)
        logger.info("Starting Gradio E2E Tests")
        logger.info("=" * 80)

        # Run tests in sequence
        tests = [
            ("Server Health", self.test_server_health),
            ("Text Analysis", self.test_text_analysis),
            ("Document Analysis", self.test_document_analysis),
            ("Image Analysis", self.test_image_analysis),
            ("Transaction Monitoring", self.test_transaction_monitoring),
            ("Compliance Checks", self.test_compliance_checks),
        ]

        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name}...")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                self.log_result(test_name, False, f"Test crashed: {e}")

            # Small delay between tests
            await asyncio.sleep(1)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("E2E Test Summary")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {self.passed + self.failed}")
        logger.info(f"Passed: {self.passed} ✅")
        logger.info(f"Failed: {self.failed} ❌")
        logger.info(f"Pass Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")

        # Print detailed results
        logger.info("\nDetailed Results:")
        for result in self.results:
            logger.info(f"  {result['status']} {result['test']}")
            if not result["passed"] and result["details"]:
                logger.info(f"      Details: {result['details']}")

        # Save results to file
        results_path = Path("e2e_test_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": self.passed + self.failed,
                        "passed": self.passed,
                        "failed": self.failed,
                        "pass_rate": (
                            self.passed / (self.passed + self.failed) * 100
                            if (self.passed + self.failed) > 0
                            else 0
                        ),
                    },
                    "results": self.results,
                },
                f,
                indent=2,
            )

        logger.info(f"\nResults saved to {results_path}")

        return self.failed == 0


async def main():
    """Main test runner."""
    tester = GradioE2ETester()

    # Wait a bit for server to be ready
    logger.info("Waiting for server to be fully ready...")
    await asyncio.sleep(2)

    # Run all tests
    success = await tester.run_all_tests()

    if success:
        logger.success("\n🎉 All E2E tests passed successfully!")
        return 0
    else:
        logger.error("\n❌ Some E2E tests failed. Please review the results.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
