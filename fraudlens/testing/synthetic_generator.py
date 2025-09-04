"""
Synthetic fraud data generator for testing.

Author: Yobie Benjamin
Date: 2025
"""

import json
import random
import string
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


@dataclass
class SyntheticDocument:
    """Synthetic document data."""

    doc_type: str
    content: str
    metadata: Dict[str, Any]
    file_path: Optional[Path] = None
    fraud_indicators: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_type": self.doc_type,
            "content": self.content,
            "metadata": self.metadata,
            "file_path": str(self.file_path) if self.file_path else None,
            "fraud_indicators": self.fraud_indicators or [],
        }


@dataclass
class SyntheticEmail:
    """Synthetic email data."""

    sender: str
    recipient: str
    subject: str
    body: str
    headers: Dict[str, str]
    attachments: List[str]
    fraud_type: Optional[str] = None
    fraud_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "subject": self.subject,
            "body": self.body,
            "headers": self.headers,
            "attachments": self.attachments,
            "fraud_type": self.fraud_type,
            "fraud_score": self.fraud_score,
        }


@dataclass
class MultiModalCase:
    """Multi-modal fraud test case."""

    case_id: str
    text_data: Optional[str] = None
    image_data: Optional[Path] = None
    document_data: Optional[Path] = None
    audio_data: Optional[Path] = None
    metadata: Dict[str, Any] = None
    ground_truth: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "text_data": self.text_data,
            "image_data": str(self.image_data) if self.image_data else None,
            "document_data": str(self.document_data) if self.document_data else None,
            "audio_data": str(self.audio_data) if self.audio_data else None,
            "metadata": self.metadata or {},
            "ground_truth": self.ground_truth or {},
        }


class SyntheticFraudGenerator:
    """Generator for synthetic fraud data."""

    def __init__(self, output_dir: str = "synthetic_data", seed: int = 42):
        """
        Initialize generator.

        Args:
            output_dir: Directory for generated files
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Initialize Faker
        self.faker = Faker()
        Faker.seed(seed)

        # Fraud templates
        self.phishing_templates = self._load_phishing_templates()
        self.document_templates = self._load_document_templates()
        self.transaction_patterns = self._load_transaction_patterns()

    def generate_phishing_email(
        self,
        fraud_type: str = "generic",
        language: str = "en",
        urgency_level: int = 5,
        personalization: bool = False,
    ) -> SyntheticEmail:
        """
        Generate synthetic phishing email.

        Args:
            fraud_type: Type of phishing (generic, spear, whaling, etc.)
            language: Language code
            urgency_level: Urgency level (1-10)
            personalization: Add personalization elements

        Returns:
            Synthetic phishing email
        """
        # Generate recipient
        recipient = self.faker.email()
        recipient_name = self.faker.name() if personalization else "Customer"

        # Select template based on fraud type
        templates = {
            "generic": self._generate_generic_phishing,
            "spear": self._generate_spear_phishing,
            "whaling": self._generate_whaling_phishing,
            "tech_support": self._generate_tech_support_scam,
            "romance": self._generate_romance_scam,
            "lottery": self._generate_lottery_scam,
        }

        generator = templates.get(fraud_type, self._generate_generic_phishing)

        # Generate email content
        sender, subject, body = generator(recipient_name, urgency_level)

        # Add fraud indicators
        body = self._inject_fraud_indicators(body, urgency_level)

        # Generate headers
        headers = {
            "From": sender,
            "To": recipient,
            "Date": datetime.now().isoformat(),
            "Message-ID": f"<{self.faker.uuid4()}@{sender.split('@')[1] if '@' in sender else 'example.com'}>",
            "X-Mailer": random.choice(["Outlook", "Gmail", "Yahoo", "Custom Mailer"]),
            "X-Originating-IP": self.faker.ipv4(),
        }

        # Add suspicious headers for high urgency
        if urgency_level > 7:
            headers["X-Priority"] = "1 (Highest)"
            headers["Importance"] = "High"

        email = SyntheticEmail(
            sender=sender,
            recipient=recipient,
            subject=subject,
            body=body,
            headers=headers,
            attachments=[],
            fraud_type=fraud_type,
            fraud_score=min(1.0, urgency_level / 10.0 + random.uniform(0, 0.2)),
        )

        return email

    def _generate_generic_phishing(
        self,
        recipient_name: str,
        urgency_level: int,
    ) -> Tuple[str, str, str]:
        """Generate generic phishing email."""
        companies = ["PayPal", "Amazon", "Netflix", "Apple", "Microsoft", "Google"]
        company = random.choice(companies)

        # Typosquatted sender
        sender_domain = (
            company.lower().replace("a", "@") if urgency_level > 7 else f"{company.lower()}.com"
        )
        sender = f"security@{sender_domain}"

        subjects = [
            f"Urgent: Your {company} account requires verification",
            f"Action Required: Suspicious activity on your {company} account",
            f"Important: Update your {company} payment information",
            f"Security Alert: Your {company} account will be suspended",
        ]

        subject = random.choice(subjects)

        body = f"""Dear {recipient_name},

We have detected suspicious activity on your {company} account. For your security, we have temporarily limited access to your account.

To restore full access, please verify your account immediately:

[Click here to verify your account]

If you do not verify within 24 hours, your account will be permanently suspended.

Thank you for your cooperation.

{company} Security Team"""

        return sender, subject, body

    def _generate_spear_phishing(
        self,
        recipient_name: str,
        urgency_level: int,
    ) -> Tuple[str, str, str]:
        """Generate spear phishing email."""
        # Simulate knowledge of recipient
        company = self.faker.company()
        ceo_name = self.faker.name()

        sender = f"{ceo_name.lower().replace(' ', '.')}@{company.lower().replace(' ', '')}.com"
        subject = f"Urgent: Wire Transfer Required - {ceo_name}"

        body = f"""Hi {recipient_name},

I need you to process an urgent wire transfer for a confidential acquisition. I'm currently in meetings and cannot call.

Please transfer $45,000 to the following account immediately:

Bank: International Business Bank
Account: 9876543210
SWIFT: IBBUS33XXX
Reference: ACQ-2024-CONF

This is time-sensitive and confidential. Do not discuss with anyone else.

Send confirmation once complete.

{ceo_name}
CEO, {company}

Sent from my iPhone"""

        return sender, subject, body

    def _generate_whaling_phishing(
        self,
        recipient_name: str,
        urgency_level: int,
    ) -> Tuple[str, str, str]:
        """Generate whaling (CEO fraud) email."""
        sender = "legal@tax-authority.gov"
        subject = "Notice of Tax Audit - Immediate Response Required"

        body = f"""Dear {recipient_name},

This is an official notice from the Tax Authority regarding your company's tax filings for the fiscal year 2023-2024.

Our audit has revealed discrepancies that require immediate clarification. Failure to respond within 48 hours will result in:

• Immediate asset freeze
• Criminal investigation
• Public disclosure of audit findings

To avoid these consequences, please provide the following documents:

1. Complete financial statements for 2023-2024
2. All bank account details
3. Employee payroll records
4. Tax identification documents

Submit all documents through our secure portal:
[SECURE UPLOAD PORTAL]

Case Reference: TX-2024-{random.randint(10000, 99999)}

Sincerely,
Tax Compliance Department"""

        return sender, subject, body

    def _generate_tech_support_scam(
        self,
        recipient_name: str,
        urgency_level: int,
    ) -> Tuple[str, str, str]:
        """Generate tech support scam email."""
        sender = "support@windows-security.com"
        subject = "Critical Security Alert: Virus Detected on Your Computer"

        body = f"""SECURITY WARNING!

Dear {recipient_name},

Our automated security scan has detected critical viruses on your computer:

• Trojan.Win32.Generic
• Malware.Ransomware.Crypto
• Spyware.KeyLogger.X

Your personal data, passwords, and files are at risk!

IMMEDIATE ACTION REQUIRED:

Call our 24/7 support hotline: 1-800-TECH-FIX
Or allow remote access for immediate virus removal:
[DOWNLOAD REMOTE ACCESS TOOL]

Warning: Every minute counts. Delaying action may result in complete data loss.

Microsoft Certified Support Team

This is an automated security notification. Do not ignore this message."""

        return sender, subject, body

    def _generate_romance_scam(
        self,
        recipient_name: str,
        urgency_level: int,
    ) -> Tuple[str, str, str]:
        """Generate romance scam email."""
        scammer_name = self.faker.name()
        sender = f"{scammer_name.lower().replace(' ', '_')}@loveconnect.com"
        subject = f"My dearest {recipient_name} - I need your help"

        body = f"""My dearest {recipient_name},

I know we have only been talking for a few weeks, but I feel such a strong connection with you. You have brought light into my life during these dark times.

I am writing with a heavy heart. I am currently stuck in {self.faker.country()} due to a medical emergency. The hospital is demanding $5,000 for my treatment and they won't let me leave without payment.

I promise to pay you back as soon as I return home. I have substantial savings but cannot access them from here.

Please, my love, can you help me? I need you to send money via Western Union to:

Name: {scammer_name}
Location: {self.faker.city()}, {self.faker.country()}
Amount: $5,000

I cannot wait to finally meet you in person and hold you in my arms.

With all my love,
{scammer_name}"""

        return sender, subject, body

    def _generate_lottery_scam(
        self,
        recipient_name: str,
        urgency_level: int,
    ) -> Tuple[str, str, str]:
        """Generate lottery scam email."""
        sender = "claims@international-lottery.org"
        subject = "Congratulations! You've Won $10,000,000"

        amount = random.choice(["$10,000,000", "€5,000,000", "£7,500,000"])
        reference = f"WIN/{random.randint(100000, 999999)}/INT"

        body = f"""CONGRATULATIONS {recipient_name.upper()}!

We are pleased to inform you that you have won {amount} in the International Online Lottery!

Your email was randomly selected from millions of active email addresses worldwide.

Winning Details:
Reference Number: {reference}
Winning Amount: {amount}
Draw Date: {datetime.now().date()}

To claim your prize, you must:

1. Pay the processing fee of $500
2. Provide proof of identity
3. Complete the claim form within 7 days

Failure to claim within the deadline will result in forfeiture of your winnings.

Contact our claims agent immediately:
Email: agent@lottery-claims.net
Phone: +{random.randint(10, 99)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}

Congratulations once again!

International Lottery Commission"""

        return sender, subject, body

    def _inject_fraud_indicators(self, body: str, urgency_level: int) -> str:
        """Inject fraud indicators into email body."""
        indicators = []

        if urgency_level > 7:
            indicators.extend(
                [
                    "\n\n⚠️ This email will self-destruct in 24 hours",
                    "\n\nACT NOW - Limited time offer!",
                    "\n\n*** FINAL NOTICE ***",
                ]
            )

        if urgency_level > 5:
            # Add suspicious URLs
            body = body.replace(
                "[Click here",
                "[Click here: http://bit.ly/" + "".join(random.choices(string.ascii_letters, k=7)),
            )

            # Add urgency words
            urgency_words = ["URGENT", "IMMEDIATE", "ACT NOW", "FINAL NOTICE"]
            body = random.choice(urgency_words) + ": " + body

        # Add typos for realism
        if random.random() > 0.7:
            typos = [
                ("your", "you're"),
                ("account", "acount"),
                ("verify", "varify"),
                ("security", "secutiry"),
            ]
            for correct, typo in typos:
                if random.random() > 0.5 and correct in body:
                    body = body.replace(correct, typo, 1)

        return body + "".join(indicators)

    def create_forged_document(
        self,
        doc_type: str = "invoice",
        forgery_type: str = "altered",
        output_format: str = "pdf",
    ) -> SyntheticDocument:
        """
        Create forged document for testing.

        Args:
            doc_type: Type of document (invoice, contract, id, etc.)
            forgery_type: Type of forgery (altered, fake, cloned)
            output_format: Output format (pdf, image)

        Returns:
            Synthetic forged document
        """
        generators = {
            "invoice": self._create_forged_invoice,
            "contract": self._create_forged_contract,
            "id": self._create_forged_id,
            "bank_statement": self._create_forged_bank_statement,
            "certificate": self._create_forged_certificate,
        }

        generator = generators.get(doc_type, self._create_forged_invoice)
        document = generator(forgery_type)

        # Generate file
        if output_format == "pdf":
            file_path = self._generate_pdf(document)
        else:
            file_path = self._generate_image(document)

        document.file_path = file_path

        return document

    def _create_forged_invoice(self, forgery_type: str) -> SyntheticDocument:
        """Create forged invoice."""
        company = self.faker.company()
        invoice_num = f"INV-{random.randint(10000, 99999)}"

        # Original content
        items = []
        for _ in range(random.randint(3, 7)):
            items.append(
                {
                    "description": self.faker.bs(),
                    "quantity": random.randint(1, 100),
                    "price": round(random.uniform(10, 1000), 2),
                }
            )

        subtotal = sum(item["quantity"] * item["price"] for item in items)

        content = f"""
INVOICE

{company}
{self.faker.address()}

Invoice Number: {invoice_num}
Date: {datetime.now().date()}
Due Date: {(datetime.now() + timedelta(days=30)).date()}

Bill To:
{self.faker.name()}
{self.faker.company()}
{self.faker.address()}

Items:
"""

        for item in items:
            content += f"\n{item['description']} - Qty: {item['quantity']} x ${item['price']} = ${item['quantity'] * item['price']:.2f}"

        # Apply forgery
        fraud_indicators = []

        if forgery_type == "altered":
            # Alter amounts
            content += f"\n\nSubtotal: ${subtotal:.2f}"
            content += f"\nTax (10%): ${subtotal * 0.1:.2f}"
            altered_total = subtotal * 1.5  # Inflate by 50%
            content += f"\nTotal: ${altered_total:.2f}"
            fraud_indicators.append("amount_tampering")
        elif forgery_type == "fake":
            # Completely fabricated
            content += f"\n\nTotal: ${random.uniform(10000, 100000):.2f}"
            fraud_indicators.append("fabricated_document")
        else:
            content += f"\n\nTotal: ${subtotal * 1.1:.2f}"

        # Add subtle forgery indicators
        if random.random() > 0.5:
            content = content.replace("Invoice", "Invioce")  # Typo
            fraud_indicators.append("suspicious_typo")

        metadata = {
            "doc_type": "invoice",
            "company": company,
            "invoice_number": invoice_num,
            "forgery_type": forgery_type,
            "created": datetime.now().isoformat(),
        }

        return SyntheticDocument(
            doc_type="invoice",
            content=content,
            metadata=metadata,
            fraud_indicators=fraud_indicators,
        )

    def _create_forged_contract(self, forgery_type: str) -> SyntheticDocument:
        """Create forged contract."""
        party1 = self.faker.company()
        party2 = self.faker.company()

        content = f"""
SERVICE AGREEMENT

This Agreement is entered into as of {datetime.now().date()} between:

Party 1: {party1}
Address: {self.faker.address()}

Party 2: {party2}
Address: {self.faker.address()}

SERVICES:
Party 2 agrees to provide {self.faker.bs()} services to Party 1.

PAYMENT:
Party 1 agrees to pay Party 2 the amount of ${random.uniform(10000, 100000):.2f}.

TERM:
This agreement shall commence on {datetime.now().date()} and continue for a period of 12 months.
"""

        fraud_indicators = []

        if forgery_type == "altered":
            # Alter signature area
            content += "\n\nSIGNATURES:\n_______________ (Party 1)\n_______________ (Party 2)"
            content += "\n\n[Digital signature altered after signing]"
            fraud_indicators.append("signature_tampering")
        elif forgery_type == "fake":
            content += "\n\n[Forged signatures applied]"
            fraud_indicators.append("forged_signatures")

        metadata = {
            "doc_type": "contract",
            "parties": [party1, party2],
            "forgery_type": forgery_type,
        }

        return SyntheticDocument(
            doc_type="contract",
            content=content,
            metadata=metadata,
            fraud_indicators=fraud_indicators,
        )

    def _create_forged_id(self, forgery_type: str) -> SyntheticDocument:
        """Create forged ID document."""
        name = self.faker.name()
        dob = self.faker.date_of_birth(minimum_age=18, maximum_age=80)
        id_number = "".join(random.choices(string.digits, k=9))

        content = f"""
IDENTIFICATION DOCUMENT

Name: {name}
Date of Birth: {dob}
ID Number: {id_number}
Issue Date: {datetime.now().date() - timedelta(days=random.randint(30, 1000))}
Expiry Date: {datetime.now().date() + timedelta(days=random.randint(365, 1825))}

Address:
{self.faker.address()}
"""

        fraud_indicators = []

        if forgery_type == "altered":
            # Alter DOB to appear younger/older
            content += "\n\n[Date of birth appears modified]"
            fraud_indicators.append("dob_alteration")
        elif forgery_type == "fake":
            content += "\n\n[Hologram and security features missing]"
            fraud_indicators.append("missing_security_features")

        metadata = {
            "doc_type": "id",
            "name": name,
            "id_number": id_number,
            "forgery_type": forgery_type,
        }

        return SyntheticDocument(
            doc_type="id",
            content=content,
            metadata=metadata,
            fraud_indicators=fraud_indicators,
        )

    def _create_forged_bank_statement(self, forgery_type: str) -> SyntheticDocument:
        """Create forged bank statement."""
        bank_name = random.choice(["Chase Bank", "Bank of America", "Wells Fargo", "Citi Bank"])
        account_number = "".join(random.choices(string.digits, k=10))

        content = f"""
{bank_name}
BANK STATEMENT

Account Holder: {self.faker.name()}
Account Number: ****{account_number[-4:]}
Statement Period: {(datetime.now() - timedelta(days=30)).date()} to {datetime.now().date()}

Opening Balance: ${random.uniform(1000, 10000):.2f}

Transactions:
"""

        transactions = []
        balance = random.uniform(1000, 10000)

        for i in range(random.randint(10, 20)):
            date = datetime.now() - timedelta(days=random.randint(1, 30))
            amount = random.uniform(-500, 1000)
            balance += amount

            transactions.append(
                {
                    "date": date.date(),
                    "description": self.faker.company() if amount < 0 else "Deposit",
                    "amount": amount,
                    "balance": balance,
                }
            )

        for trans in sorted(transactions, key=lambda x: x["date"]):
            content += f"\n{trans['date']} | {trans['description'][:30]:30} | ${trans['amount']:>10.2f} | ${trans['balance']:>10.2f}"

        fraud_indicators = []

        if forgery_type == "altered":
            # Inflate final balance
            balance *= 10
            content += f"\n\nClosing Balance: ${balance:.2f}"
            fraud_indicators.append("balance_manipulation")
        elif forgery_type == "fake":
            content += f"\n\nClosing Balance: ${random.uniform(100000, 1000000):.2f}"
            fraud_indicators.append("fabricated_transactions")
        else:
            content += f"\n\nClosing Balance: ${balance:.2f}"

        metadata = {
            "doc_type": "bank_statement",
            "bank": bank_name,
            "account_number": account_number,
            "forgery_type": forgery_type,
        }

        return SyntheticDocument(
            doc_type="bank_statement",
            content=content,
            metadata=metadata,
            fraud_indicators=fraud_indicators,
        )

    def _create_forged_certificate(self, forgery_type: str) -> SyntheticDocument:
        """Create forged certificate."""
        institution = self.faker.company() + " University"
        recipient = self.faker.name()
        degree = random.choice(["Bachelor of Science", "Master of Arts", "Doctor of Philosophy"])
        field = random.choice(
            ["Computer Science", "Business Administration", "Engineering", "Medicine"]
        )

        content = f"""
{institution}

This is to certify that

{recipient}

has successfully completed all requirements for the degree of

{degree}
in
{field}

Awarded on: {self.faker.date_between(start_date='-5y', end_date='today')}

Dean: {self.faker.name()}
President: {self.faker.name()}
"""

        fraud_indicators = []

        if forgery_type == "altered":
            content += "\n\n[Degree type appears modified]"
            fraud_indicators.append("degree_alteration")
        elif forgery_type == "fake":
            content += "\n\n[Institution not accredited]"
            fraud_indicators.append("fake_institution")

        metadata = {
            "doc_type": "certificate",
            "institution": institution,
            "recipient": recipient,
            "degree": degree,
            "forgery_type": forgery_type,
        }

        return SyntheticDocument(
            doc_type="certificate",
            content=content,
            metadata=metadata,
            fraud_indicators=fraud_indicators,
        )

    def _generate_pdf(self, document: SyntheticDocument) -> Path:
        """Generate PDF file from document."""
        filename = f"{document.doc_type}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        file_path = self.output_dir / filename

        c = canvas.Canvas(str(file_path), pagesize=letter)

        # Add content
        y_position = 750
        for line in document.content.split("\n"):
            if y_position < 50:
                c.showPage()
                y_position = 750
            c.drawString(50, y_position, line)
            y_position -= 15

        # Add watermark for forged documents
        if document.fraud_indicators:
            c.setFont("Helvetica", 60)
            c.setFillColorRGB(0.9, 0.9, 0.9)
            c.saveState()
            c.translate(300, 400)
            c.rotate(45)
            c.drawCentredString(0, 0, "TEST DOCUMENT")
            c.restoreState()

        c.save()

        return file_path

    def _generate_image(self, document: SyntheticDocument) -> Path:
        """Generate image file from document."""
        filename = f"{document.doc_type}_{datetime.now():%Y%m%d_%H%M%S}.png"
        file_path = self.output_dir / filename

        # Create image
        img = Image.new("RGB", (800, 1000), color="white")
        draw = ImageDraw.Draw(img)

        # Add content
        y_position = 50
        for line in document.content.split("\n"):
            draw.text((50, y_position), line, fill="black")
            y_position += 20

        # Add artifacts for forged documents
        if document.fraud_indicators:
            # Add noise
            pixels = img.load()
            for _ in range(1000):
                x = random.randint(0, 799)
                y = random.randint(0, 999)
                pixels[x, y] = (
                    random.randint(200, 255),
                    random.randint(200, 255),
                    random.randint(200, 255),
                )

        img.save(file_path)

        return file_path

    def synthesize_fraud_scenario(
        self,
        scenario_type: str = "phishing_campaign",
        complexity: str = "medium",
    ) -> MultiModalCase:
        """
        Synthesize complete fraud scenario.

        Args:
            scenario_type: Type of fraud scenario
            complexity: Complexity level (low, medium, high)

        Returns:
            Multi-modal fraud case
        """
        case_id = f"CASE_{datetime.now():%Y%m%d_%H%M%S}_{random.randint(1000, 9999)}"

        scenarios = {
            "phishing_campaign": self._scenario_phishing_campaign,
            "identity_theft": self._scenario_identity_theft,
            "document_fraud": self._scenario_document_fraud,
            "financial_fraud": self._scenario_financial_fraud,
            "social_engineering": self._scenario_social_engineering,
        }

        generator = scenarios.get(scenario_type, self._scenario_phishing_campaign)
        case = generator(case_id, complexity)

        return case

    def _scenario_phishing_campaign(
        self,
        case_id: str,
        complexity: str,
    ) -> MultiModalCase:
        """Generate phishing campaign scenario."""
        # Generate email
        email = self.generate_phishing_email(
            fraud_type="spear" if complexity == "high" else "generic",
            urgency_level=8 if complexity == "high" else 5,
        )

        # Generate fake landing page screenshot
        landing_page = self._create_fake_landing_page()

        # Create supporting document
        if complexity in ["medium", "high"]:
            document = self.create_forged_document(
                doc_type="invoice",
                forgery_type="fake",
            )
        else:
            document = None

        ground_truth = {
            "is_fraud": True,
            "fraud_type": "phishing",
            "severity": complexity,
            "indicators": [
                "suspicious_sender",
                "urgency_pressure",
                "fake_landing_page",
            ],
        }

        return MultiModalCase(
            case_id=case_id,
            text_data=email.body,
            image_data=landing_page,
            document_data=document.file_path if document else None,
            metadata={
                "email": email.to_dict(),
                "complexity": complexity,
            },
            ground_truth=ground_truth,
        )

    def _scenario_identity_theft(
        self,
        case_id: str,
        complexity: str,
    ) -> MultiModalCase:
        """Generate identity theft scenario."""
        # Create forged ID
        fake_id = self.create_forged_document(
            doc_type="id",
            forgery_type="fake" if complexity == "high" else "altered",
        )

        # Create supporting documents
        bank_statement = self.create_forged_document(
            doc_type="bank_statement",
            forgery_type="altered",
        )

        # Generate application text
        application_text = f"""
Loan Application

Name: {self.faker.name()}
SSN: XXX-XX-{random.randint(1000, 9999)}
Income: ${random.randint(50000, 150000)}
Requested Amount: ${random.randint(10000, 50000)}

Purpose: {self.faker.bs()}
"""

        ground_truth = {
            "is_fraud": True,
            "fraud_type": "identity_theft",
            "severity": complexity,
            "indicators": [
                "forged_id",
                "altered_documents",
                "inconsistent_information",
            ],
        }

        return MultiModalCase(
            case_id=case_id,
            text_data=application_text,
            image_data=fake_id.file_path,
            document_data=bank_statement.file_path,
            metadata={
                "id_doc": fake_id.to_dict(),
                "bank_statement": bank_statement.to_dict(),
            },
            ground_truth=ground_truth,
        )

    def _scenario_document_fraud(
        self,
        case_id: str,
        complexity: str,
    ) -> MultiModalCase:
        """Generate document fraud scenario."""
        # Create multiple forged documents
        docs = []
        doc_types = ["invoice", "contract", "certificate"] if complexity == "high" else ["invoice"]

        for doc_type in doc_types:
            doc = self.create_forged_document(
                doc_type=doc_type,
                forgery_type="altered" if complexity == "medium" else "fake",
            )
            docs.append(doc)

        ground_truth = {
            "is_fraud": True,
            "fraud_type": "document_fraud",
            "severity": complexity,
            "indicators": [doc.fraud_indicators for doc in docs],
        }

        return MultiModalCase(
            case_id=case_id,
            text_data=docs[0].content,
            document_data=docs[0].file_path,
            metadata={
                "documents": [doc.to_dict() for doc in docs],
                "complexity": complexity,
            },
            ground_truth=ground_truth,
        )

    def _scenario_financial_fraud(
        self,
        case_id: str,
        complexity: str,
    ) -> MultiModalCase:
        """Generate financial fraud scenario."""
        # Generate transaction data
        transactions = self.generate_transaction_pattern(
            pattern_type="money_laundering" if complexity == "high" else "card_fraud",
            num_transactions=100 if complexity == "high" else 20,
        )

        # Create fake receipts
        receipt = self._create_fake_receipt()

        ground_truth = {
            "is_fraud": True,
            "fraud_type": "financial_fraud",
            "severity": complexity,
            "indicators": [
                "suspicious_transactions",
                "velocity_anomaly",
                "amount_anomaly",
            ],
        }

        return MultiModalCase(
            case_id=case_id,
            text_data=json.dumps(transactions, indent=2),
            image_data=receipt,
            metadata={
                "transaction_count": len(transactions),
                "total_amount": sum(t.get("amount", 0) for t in transactions),
            },
            ground_truth=ground_truth,
        )

    def _scenario_social_engineering(
        self,
        case_id: str,
        complexity: str,
    ) -> MultiModalCase:
        """Generate social engineering scenario."""
        # Generate conversation transcript
        transcript = self._generate_social_engineering_transcript(complexity)

        # Generate fake social media profile
        profile_image = self._create_fake_profile()

        ground_truth = {
            "is_fraud": True,
            "fraud_type": "social_engineering",
            "severity": complexity,
            "indicators": [
                "trust_exploitation",
                "information_gathering",
                "urgency_creation",
            ],
        }

        return MultiModalCase(
            case_id=case_id,
            text_data=transcript,
            image_data=profile_image,
            metadata={
                "attack_vector": "social_media",
                "complexity": complexity,
            },
            ground_truth=ground_truth,
        )

    def _generate_social_engineering_transcript(self, complexity: str) -> str:
        """Generate social engineering conversation."""
        attacker = self.faker.name()
        victim = self.faker.name()

        transcript = f"""
Chat Transcript
Participants: {attacker}, {victim}

{attacker}: Hi {victim}! I'm from IT support. We've detected unusual activity on your account.
{victim}: Oh no! What kind of activity?
{attacker}: Someone attempted to log in from Russia. Did you recently travel there?
{victim}: No, I haven't left the country.
{attacker}: We need to secure your account immediately. Can you verify your current password?
"""

        if complexity == "high":
            transcript += f"""
{victim}: Isn't it against policy to ask for passwords?
{attacker}: You're right to be cautious! We use a secure verification system. 
{attacker}: I'll send you a verification code. Please read it back to me.
{attacker}: [Sends fake 2FA prompt]
{victim}: The code is 453829
{attacker}: Perfect. Now I need to verify your identity. What's your employee ID?
{victim}: It's EMP78234
{attacker}: Great. For the final step, please install our security patch: [malicious link]
"""

        return transcript

    def _create_fake_landing_page(self) -> Path:
        """Create fake phishing landing page screenshot."""
        filename = f"landing_page_{datetime.now():%Y%m%d_%H%M%S}.png"
        file_path = self.output_dir / filename

        # Create image
        img = Image.new("RGB", (1200, 800), color="white")
        draw = ImageDraw.Draw(img)

        # Draw fake login form
        draw.rectangle([400, 200, 800, 600], outline="gray", width=2)
        draw.text((550, 220), "PayPaI Login", fill="blue", font=None)  # Note the typo
        draw.text((450, 300), "Email:", fill="black")
        draw.rectangle([450, 320, 750, 350], outline="gray")
        draw.text((450, 380), "Password:", fill="black")
        draw.rectangle([450, 400, 750, 430], outline="gray")
        draw.rectangle([520, 480, 680, 520], fill="blue")
        draw.text((570, 490), "Sign In", fill="white")

        # Add suspicious URL
        draw.text((450, 700), "https://paypaI-secure.fake-site.com/login", fill="gray")

        img.save(file_path)

        return file_path

    def _create_fake_receipt(self) -> Path:
        """Create fake receipt image."""
        filename = f"receipt_{datetime.now():%Y%m%d_%H%M%S}.png"
        file_path = self.output_dir / filename

        # Create image
        img = Image.new("RGB", (400, 600), color="white")
        draw = ImageDraw.Draw(img)

        # Draw receipt
        store = self.faker.company()
        draw.text((150, 50), store, fill="black")
        draw.text((50, 100), f"Date: {datetime.now().date()}", fill="black")
        draw.text((50, 130), f"Trans #: {random.randint(10000, 99999)}", fill="black")

        y = 180
        total = 0
        for _ in range(random.randint(3, 7)):
            item = self.faker.word()
            price = random.uniform(10, 100)
            draw.text((50, y), f"{item[:20]:20} ${price:.2f}", fill="black")
            y += 30
            total += price

        draw.line([(50, y), (350, y)], fill="black")
        y += 20
        draw.text((50, y), f"TOTAL: ${total:.2f}", fill="black")

        img.save(file_path)

        return file_path

    def _create_fake_profile(self) -> Path:
        """Create fake social media profile image."""
        filename = f"profile_{datetime.now():%Y%m%d_%H%M%S}.png"
        file_path = self.output_dir / filename

        # Create image
        img = Image.new("RGB", (200, 200), color="lightgray")
        draw = ImageDraw.Draw(img)

        # Draw placeholder profile
        draw.ellipse([50, 50, 150, 150], fill="white", outline="gray")
        draw.text((85, 85), "FAKE", fill="red")

        img.save(file_path)

        return file_path

    def generate_transaction_pattern(
        self,
        pattern_type: str = "normal",
        num_transactions: int = 100,
        anomaly_rate: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Generate transaction patterns.

        Args:
            pattern_type: Type of pattern (normal, card_fraud, money_laundering)
            num_transactions: Number of transactions
            anomaly_rate: Rate of anomalous transactions

        Returns:
            List of transaction records
        """
        transactions = []
        current_balance = random.uniform(1000, 10000)

        for i in range(num_transactions):
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )

            if pattern_type == "normal":
                transaction = self._generate_normal_transaction(timestamp, current_balance)
            elif pattern_type == "card_fraud":
                if random.random() < anomaly_rate:
                    transaction = self._generate_fraud_transaction(timestamp, current_balance)
                else:
                    transaction = self._generate_normal_transaction(timestamp, current_balance)
            elif pattern_type == "money_laundering":
                transaction = self._generate_ml_transaction(timestamp, current_balance, i)
            else:
                transaction = self._generate_normal_transaction(timestamp, current_balance)

            current_balance += transaction["amount"]
            transaction["balance"] = current_balance
            transactions.append(transaction)

        return sorted(transactions, key=lambda x: x["timestamp"])

    def _generate_normal_transaction(
        self,
        timestamp: datetime,
        balance: float,
    ) -> Dict[str, Any]:
        """Generate normal transaction."""
        merchants = [
            "Grocery Store",
            "Gas Station",
            "Restaurant",
            "Online Shopping",
            "Utility Company",
            "Coffee Shop",
        ]

        return {
            "id": str(random.randint(100000, 999999)),
            "timestamp": timestamp.isoformat(),
            "merchant": random.choice(merchants),
            "amount": -random.uniform(10, 200),
            "type": "purchase",
            "location": self.faker.city(),
            "fraud_score": random.uniform(0, 0.3),
        }

    def _generate_fraud_transaction(
        self,
        timestamp: datetime,
        balance: float,
    ) -> Dict[str, Any]:
        """Generate fraudulent transaction."""
        # Unusual merchants
        merchants = [
            "Online Casino",
            "Crypto Exchange",
            "Foreign ATM",
            "Luxury Goods Store",
            "Wire Transfer Service",
        ]

        # Unusual patterns
        amount = -random.uniform(500, min(balance * 0.5, 5000))

        return {
            "id": str(random.randint(100000, 999999)),
            "timestamp": timestamp.isoformat(),
            "merchant": random.choice(merchants),
            "amount": amount,
            "type": "suspicious",
            "location": self.faker.country(),  # Foreign location
            "fraud_score": random.uniform(0.7, 1.0),
        }

    def _generate_ml_transaction(
        self,
        timestamp: datetime,
        balance: float,
        index: int,
    ) -> Dict[str, Any]:
        """Generate money laundering pattern transaction."""
        # Structuring pattern
        if index % 10 < 3:
            # Deposits just under reporting threshold
            amount = random.uniform(9000, 9999)
            trans_type = "deposit"
        elif index % 10 < 6:
            # Rapid transfers
            amount = -random.uniform(1000, 5000)
            trans_type = "transfer"
        else:
            # Normal transaction for cover
            return self._generate_normal_transaction(timestamp, balance)

        return {
            "id": str(random.randint(100000, 999999)),
            "timestamp": timestamp.isoformat(),
            "merchant": "Wire Transfer" if trans_type == "transfer" else "Cash Deposit",
            "amount": amount,
            "type": trans_type,
            "location": self.faker.city(),
            "fraud_score": random.uniform(0.5, 0.9),
        }

    def generate_test_dataset(
        self,
        size: int = 1000,
        fraud_rate: float = 0.1,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete test dataset.

        Args:
            size: Number of samples
            fraud_rate: Percentage of fraud cases
            output_file: Path to save dataset

        Returns:
            Test dataset
        """
        dataset = {
            "metadata": {
                "size": size,
                "fraud_rate": fraud_rate,
                "created": datetime.now().isoformat(),
                "generator_version": "1.0.0",
            },
            "samples": [],
        }

        num_fraud = int(size * fraud_rate)
        num_normal = size - num_fraud

        print(f"Generating {size} samples ({num_fraud} fraud, {num_normal} normal)...")

        # Generate fraud samples
        for i in range(num_fraud):
            scenario_type = random.choice(
                [
                    "phishing_campaign",
                    "identity_theft",
                    "document_fraud",
                    "financial_fraud",
                    "social_engineering",
                ]
            )

            complexity = random.choice(["low", "medium", "high"])

            case = self.synthesize_fraud_scenario(scenario_type, complexity)

            sample = {
                "id": f"SAMPLE_{i:05d}",
                "type": "fraud",
                "scenario": scenario_type,
                "data": case.to_dict(),
            }

            dataset["samples"].append(sample)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_fraud} fraud samples")

        # Generate normal samples
        for i in range(num_normal):
            # Create benign data
            sample = {
                "id": f"SAMPLE_{num_fraud + i:05d}",
                "type": "normal",
                "scenario": "legitimate",
                "data": {
                    "text_data": self.faker.text(max_nb_chars=500),
                    "metadata": {
                        "source": "legitimate",
                        "timestamp": datetime.now().isoformat(),
                    },
                    "ground_truth": {
                        "is_fraud": False,
                        "fraud_type": None,
                    },
                },
            }

            dataset["samples"].append(sample)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_normal} normal samples")

        # Shuffle samples
        random.shuffle(dataset["samples"])

        # Save dataset
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self.output_dir / f"test_dataset_{datetime.now():%Y%m%d_%H%M%S}.json"

        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"\nDataset saved to: {output_path}")

        return dataset

    def _load_phishing_templates(self) -> Dict[str, List[str]]:
        """Load phishing email templates."""
        return {
            "subjects": [
                "Urgent: Account Security Alert",
                "Action Required: Verify Your Account",
                "Important: Suspicious Activity Detected",
                "Final Notice: Account Suspension",
            ],
            "openings": [
                "We have detected unusual activity",
                "Your account security is at risk",
                "Immediate action is required",
                "This is an automated security alert",
            ],
            "threats": [
                "Your account will be suspended",
                "Access will be permanently revoked",
                "Legal action may be taken",
                "Your funds will be frozen",
            ],
            "actions": [
                "Click here to verify",
                "Update your information",
                "Confirm your identity",
                "Secure your account",
            ],
        }

    def _load_document_templates(self) -> Dict[str, Any]:
        """Load document templates."""
        return {
            "invoice": {
                "fields": ["invoice_number", "date", "amount", "items"],
                "forgery_indicators": ["amount_mismatch", "date_alteration", "signature_missing"],
            },
            "contract": {
                "fields": ["parties", "terms", "signatures", "date"],
                "forgery_indicators": ["signature_forgery", "terms_modified", "backdating"],
            },
            "id": {
                "fields": ["name", "dob", "id_number", "photo"],
                "forgery_indicators": [
                    "photo_replaced",
                    "dob_altered",
                    "security_features_missing",
                ],
            },
        }

    def _load_transaction_patterns(self) -> Dict[str, Any]:
        """Load transaction patterns."""
        return {
            "normal": {
                "amount_range": (10, 500),
                "frequency": "regular",
                "locations": "consistent",
            },
            "card_fraud": {
                "amount_range": (500, 5000),
                "frequency": "burst",
                "locations": "varied",
            },
            "money_laundering": {
                "amount_range": (9000, 9999),
                "frequency": "structured",
                "locations": "multiple",
            },
        }


if __name__ == "__main__":
    # Example usage
    generator = SyntheticFraudGenerator()

    # Generate phishing email
    email = generator.generate_phishing_email(
        fraud_type="spear",
        urgency_level=8,
        personalization=True,
    )
    print(f"\nGenerated Phishing Email:")
    print(f"Subject: {email.subject}")
    print(f"Fraud Score: {email.fraud_score:.2f}")

    # Create forged document
    document = generator.create_forged_document(
        doc_type="invoice",
        forgery_type="altered",
    )
    print(f"\nGenerated Forged Document:")
    print(f"Type: {document.doc_type}")
    print(f"Fraud Indicators: {document.fraud_indicators}")

    # Synthesize complete scenario
    scenario = generator.synthesize_fraud_scenario(
        scenario_type="phishing_campaign",
        complexity="high",
    )
    print(f"\nGenerated Fraud Scenario:")
    print(f"Case ID: {scenario.case_id}")
    print(f"Ground Truth: {scenario.ground_truth}")

    # Generate test dataset
    dataset = generator.generate_test_dataset(
        size=100,
        fraud_rate=0.2,
    )
    print(f"\nGenerated Test Dataset:")
    print(f"Total Samples: {len(dataset['samples'])}")
    print(f"Fraud Samples: {sum(1 for s in dataset['samples'] if s['type'] == 'fraud')}")
    print(f"Normal Samples: {sum(1 for s in dataset['samples'] if s['type'] == 'normal')}")
