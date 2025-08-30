"""
Document Validation System.

Validates identity documents including SSN, driver's licenses, passports,
and other government-issued IDs using known formats and algorithms.

Author: Yobie Benjamin
Date: 2025-08-28
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class DocumentValidator:
    """
    Validates identity documents using known formats and algorithms.
    
    Features:
    - SSN validation (US and international equivalents)
    - Driver's license format validation (all US states)
    - Passport MRZ verification (ICAO 9303 standard)
    - Credit card validation (Luhn algorithm)
    - National ID validation (multiple countries)
    """
    
    def __init__(self):
        """Initialize document validator."""
        self._init_ssn_rules()
        self._init_driver_license_formats()
        self._init_passport_rules()
        self._init_national_id_formats()
        
        logger.info("DocumentValidator initialized with validation rules for 50+ document types")
    
    def _init_ssn_rules(self):
        """Initialize SSN validation rules for multiple countries."""
        self.ssn_formats = {
            "US": {
                "format": r"^\d{3}-?\d{2}-?\d{4}$",
                "validator": self._validate_us_ssn,
                "description": "US Social Security Number"
            },
            "CA": {
                "format": r"^\d{3}-?\d{3}-?\d{3}$",
                "validator": self._validate_canadian_sin,
                "description": "Canadian Social Insurance Number"
            },
            "UK": {
                "format": r"^[A-Z]{2}\d{6}[A-Z]$",
                "validator": self._validate_uk_nino,
                "description": "UK National Insurance Number"
            },
            "FR": {
                "format": r"^[12]\d{2}(0[1-9]|1[0-2])\d{2}\d{3}\d{3}\d{2}$",
                "validator": self._validate_french_insee,
                "description": "French INSEE Number"
            },
            "DE": {
                "format": r"^\d{2}\d{6}[A-Z]\d{3}$",
                "validator": self._validate_german_steuerid,
                "description": "German Tax ID"
            },
            "SE": {
                "format": r"^\d{6}[-+]?\d{4}$",
                "validator": self._validate_swedish_personnummer,
                "description": "Swedish Personal Number (uses Luhn)"
            },
            "IN": {
                "format": r"^\d{4}\s?\d{4}\s?\d{4}$",
                "validator": self._validate_indian_aadhaar,
                "description": "Indian Aadhaar Number"
            },
            "AU": {
                "format": r"^\d{3}\s?\d{3}\s?\d{3}$",
                "validator": self._validate_australian_tfn,
                "description": "Australian Tax File Number"
            }
        }
    
    def _init_driver_license_formats(self):
        """Initialize US state driver's license formats."""
        self.dl_formats = {
            "AL": r"^\d{7,8}$",  # Alabama
            "AK": r"^\d{7}$",    # Alaska
            "AZ": r"^[A-Z]\d{8}$|^\d{9}$",  # Arizona
            "AR": r"^\d{9}$",    # Arkansas
            "CA": r"^[A-Z]\d{7}$",  # California
            "CO": r"^\d{9}$|^[A-Z]\d{3,6}$|^[A-Z]{2}\d{2,5}$",  # Colorado
            "CT": r"^\d{9}$",    # Connecticut
            "DE": r"^\d{1,7}$",  # Delaware
            "FL": r"^[A-Z]\d{12}$",  # Florida
            "GA": r"^\d{7,9}$",  # Georgia
            "HI": r"^[A-Z]\d{8}$|^\d{9}$",  # Hawaii
            "ID": r"^[A-Z]{2}\d{6}[A-Z]$|^\d{9}$",  # Idaho
            "IL": r"^[A-Z]\d{11,12}$",  # Illinois
            "IN": r"^\d{10}$|^\d{9}$",  # Indiana
            "IA": r"^\d{9}$|^\d{3}[A-Z]{2}\d{4}$",  # Iowa
            "KS": r"^[A-Z]\d{8}$|^K\d{8}$",  # Kansas
            "KY": r"^[A-Z]\d{8}$|^\d{9}$",  # Kentucky
            "LA": r"^\d{9}$",    # Louisiana
            "ME": r"^\d{7}$|^\d{8}$",  # Maine
            "MD": r"^[A-Z]\d{12}$",  # Maryland
            "MA": r"^S\d{8}$|^\d{9}$",  # Massachusetts
            "MI": r"^[A-Z]\d{12}$",  # Michigan
            "MN": r"^[A-Z]\d{12}$",  # Minnesota
            "MS": r"^\d{9}$",    # Mississippi
            "MO": r"^[A-Z]\d{5,9}$|^\d{9}$",  # Missouri
            "MT": r"^\d{13}$|^[A-Z]\d{8}$",  # Montana
            "NE": r"^[A-Z]\d{8}$",  # Nebraska
            "NV": r"^\d{10}$|^\d{12}$|^X\d{8}$",  # Nevada
            "NH": r"^\d{2}[A-Z]{3}\d{5}$",  # New Hampshire
            "NJ": r"^[A-Z]\d{14}$",  # New Jersey
            "NM": r"^\d{9}$",    # New Mexico
            "NY": r"^\d{9}$|^[A-Z]\d{18}$",  # New York
            "NC": r"^\d{1,12}$",  # North Carolina
            "ND": r"^[A-Z]{3}\d{6}$|^\d{9}$",  # North Dakota
            "OH": r"^[A-Z]{2}\d{6}$|^\d{8}$",  # Ohio
            "OK": r"^[A-Z]\d{9}$|^\d{9}$",  # Oklahoma
            "OR": r"^\d{1,9}$",  # Oregon
            "PA": r"^\d{8}$",    # Pennsylvania
            "RI": r"^\d{7}$|^V\d{6}$",  # Rhode Island
            "SC": r"^\d{5,11}$",  # South Carolina
            "SD": r"^\d{6,10}$",  # South Dakota
            "TN": r"^\d{7,9}$",  # Tennessee
            "TX": r"^\d{7,8}$",  # Texas
            "UT": r"^\d{4,10}$",  # Utah
            "VT": r"^\d{8}$|^\d{7}A$",  # Vermont
            "VA": r"^[A-Z]\d{8,11}$|^\d{9}$",  # Virginia
            "WA": r"^[A-Z\*]{7}\d{3}[A-Z\d]{2}$",  # Washington
            "WV": r"^[A-Z]{1,2}\d{5,6}$",  # West Virginia
            "WI": r"^[A-Z]\d{13}$",  # Wisconsin
            "WY": r"^\d{9}$"     # Wyoming
        }
    
    def _init_passport_rules(self):
        """Initialize passport MRZ validation rules."""
        self.mrz_formats = {
            "TD1": {  # ID cards
                "lines": 3,
                "line_length": 30,
                "check_digits": [14, 21, 28, 29]
            },
            "TD2": {  # ID cards, visas
                "lines": 2,
                "line_length": 36,
                "check_digits": [19, 27, 35]
            },
            "TD3": {  # Passports
                "lines": 2,
                "line_length": 44,
                "check_digits": [9, 19, 42, 43]
            }
        }
        
        # Valid country codes (ISO 3166-1 alpha-3)
        self.valid_country_codes = [
            "USA", "CAN", "GBR", "FRA", "DEU", "ITA", "ESP", "AUS",
            "NZL", "JPN", "CHN", "IND", "BRA", "MEX", "ARG", "RUS",
            "ZAF", "EGY", "NGA", "KEN", "ISR", "SAU", "UAE", "SGP",
            "HKG", "KOR", "TWN", "THA", "VNM", "IDN", "MYS", "PHL"
        ]
    
    def _init_national_id_formats(self):
        """Initialize national ID formats for various countries."""
        self.national_id_formats = {
            "IN_PAN": {  # Indian PAN card
                "format": r"^[A-Z]{5}\d{4}[A-Z]$",
                "description": "Indian PAN Card"
            },
            "MX_CURP": {  # Mexican CURP
                "format": r"^[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z\d]\d$",
                "description": "Mexican CURP"
            },
            "BR_CPF": {  # Brazilian CPF
                "format": r"^\d{3}\.\d{3}\.\d{3}-\d{2}$|^\d{11}$",
                "validator": self._validate_brazilian_cpf,
                "description": "Brazilian CPF"
            },
            "ZA_ID": {  # South African ID
                "format": r"^\d{13}$",
                "validator": self._validate_south_african_id,
                "description": "South African ID"
            },
            "ES_NIE": {  # Spanish NIE
                "format": r"^[XYZ]\d{7}[A-Z]$",
                "validator": self._validate_spanish_nie,
                "description": "Spanish NIE"
            }
        }
    
    def validate_ssn(self, ssn: str, country: str = "US") -> Dict[str, Any]:
        """
        Validate Social Security Number or equivalent.
        
        Args:
            ssn: SSN string
            country: Country code
            
        Returns:
            Validation result
        """
        if country not in self.ssn_formats:
            return {
                "valid": False,
                "error": f"Unknown country code: {country}",
                "country": country
            }
        
        format_info = self.ssn_formats[country]
        
        # Check format
        if not re.match(format_info["format"], ssn):
            return {
                "valid": False,
                "error": "Invalid format",
                "expected_format": format_info["description"],
                "country": country
            }
        
        # Run country-specific validator
        if "validator" in format_info:
            return format_info["validator"](ssn)
        
        return {
            "valid": True,
            "country": country,
            "type": format_info["description"]
        }
    
    def _validate_us_ssn(self, ssn: str) -> Dict[str, Any]:
        """Validate US Social Security Number."""
        # Remove hyphens
        clean_ssn = ssn.replace("-", "")
        
        # Check length
        if len(clean_ssn) != 9:
            return {"valid": False, "error": "SSN must be 9 digits"}
        
        # Invalid SSNs
        if clean_ssn == "000000000":
            return {"valid": False, "error": "Invalid SSN (all zeros)"}
        
        area = clean_ssn[:3]
        group = clean_ssn[3:5]
        serial = clean_ssn[5:]
        
        # Area 000, 666, or 900-999 are invalid
        if area == "000" or area == "666" or (900 <= int(area) <= 999):
            return {"valid": False, "error": "Invalid area number"}
        
        # Group 00 or Serial 0000 are invalid
        if group == "00" or serial == "0000":
            return {"valid": False, "error": "Invalid group or serial number"}
        
        # Known fake SSNs used in examples
        fake_ssns = ["123456789", "111111111", "222222222", "123121234"]
        if clean_ssn in fake_ssns:
            return {"valid": False, "error": "Known fake SSN"}
        
        return {
            "valid": True,
            "country": "US",
            "type": "Social Security Number",
            "area": area,
            "group": group
        }
    
    def _validate_canadian_sin(self, sin: str) -> Dict[str, Any]:
        """Validate Canadian Social Insurance Number using Luhn algorithm."""
        clean_sin = sin.replace("-", "").replace(" ", "")
        
        if len(clean_sin) != 9:
            return {"valid": False, "error": "SIN must be 9 digits"}
        
        # Apply Luhn algorithm
        if not self._luhn_check(clean_sin):
            return {"valid": False, "error": "Invalid SIN (Luhn check failed)"}
        
        return {
            "valid": True,
            "country": "CA",
            "type": "Social Insurance Number"
        }
    
    def _validate_uk_nino(self, nino: str) -> Dict[str, Any]:
        """Validate UK National Insurance Number."""
        if not re.match(r"^[A-Z]{2}\d{6}[A-Z]$", nino.upper()):
            return {"valid": False, "error": "Invalid NINO format"}
        
        # Invalid prefixes
        invalid_prefixes = ["BG", "GB", "NK", "KN", "TN", "NT", "ZZ"]
        if nino[:2].upper() in invalid_prefixes:
            return {"valid": False, "error": "Invalid NINO prefix"}
        
        return {
            "valid": True,
            "country": "UK",
            "type": "National Insurance Number"
        }
    
    def _validate_swedish_personnummer(self, pnr: str) -> Dict[str, Any]:
        """Validate Swedish Personal Number using Luhn algorithm."""
        clean_pnr = pnr.replace("-", "").replace("+", "")
        
        if len(clean_pnr) != 10:
            return {"valid": False, "error": "Personal number must be 10 digits"}
        
        # Apply Luhn algorithm
        if not self._luhn_check(clean_pnr):
            return {"valid": False, "error": "Invalid personal number (Luhn check failed)"}
        
        return {
            "valid": True,
            "country": "SE",
            "type": "Personal Number"
        }
    
    def _validate_french_insee(self, insee: str) -> Dict[str, Any]:
        """Validate French INSEE Number."""
        if len(insee) != 15:
            return {"valid": False, "error": "INSEE must be 15 digits"}
        
        # Check key (last 2 digits)
        key = int(insee[-2:])
        number = int(insee[:-2])
        calculated_key = 97 - (number % 97)
        
        if key != calculated_key:
            return {"valid": False, "error": "Invalid INSEE check key"}
        
        return {
            "valid": True,
            "country": "FR",
            "type": "INSEE Number"
        }
    
    def _validate_german_steuerid(self, steuerid: str) -> Dict[str, Any]:
        """Validate German Tax ID."""
        if not re.match(r"^\d{11}$", steuerid):
            return {"valid": False, "error": "Tax ID must be 11 digits"}
        
        # Check for duplicate digits (max 2 or 3 of same digit)
        digit_counts = {}
        for digit in steuerid:
            digit_counts[digit] = digit_counts.get(digit, 0) + 1
        
        if max(digit_counts.values()) > 3:
            return {"valid": False, "error": "Invalid Tax ID (too many duplicate digits)"}
        
        return {
            "valid": True,
            "country": "DE",
            "type": "Tax ID"
        }
    
    def _validate_indian_aadhaar(self, aadhaar: str) -> Dict[str, Any]:
        """Validate Indian Aadhaar Number."""
        clean_aadhaar = aadhaar.replace(" ", "")
        
        if len(clean_aadhaar) != 12:
            return {"valid": False, "error": "Aadhaar must be 12 digits"}
        
        # Verhoeff algorithm validation would go here
        # For now, basic validation
        if clean_aadhaar[0] in ["0", "1"]:
            return {"valid": False, "error": "Invalid Aadhaar (cannot start with 0 or 1)"}
        
        return {
            "valid": True,
            "country": "IN",
            "type": "Aadhaar Number"
        }
    
    def _validate_australian_tfn(self, tfn: str) -> Dict[str, Any]:
        """Validate Australian Tax File Number."""
        clean_tfn = tfn.replace(" ", "")
        
        if len(clean_tfn) != 9:
            return {"valid": False, "error": "TFN must be 9 digits"}
        
        # Apply weighted modulus algorithm
        weights = [1, 4, 3, 7, 5, 8, 6, 9, 10]
        total = sum(int(digit) * weight for digit, weight in zip(clean_tfn, weights))
        
        if total % 11 != 0:
            return {"valid": False, "error": "Invalid TFN (checksum failed)"}
        
        return {
            "valid": True,
            "country": "AU",
            "type": "Tax File Number"
        }
    
    def _validate_brazilian_cpf(self, cpf: str) -> Dict[str, Any]:
        """Validate Brazilian CPF."""
        # Remove formatting
        clean_cpf = re.sub(r'[^\d]', '', cpf)
        
        if len(clean_cpf) != 11:
            return {"valid": False, "error": "CPF must be 11 digits"}
        
        # Check for known invalid CPFs
        if clean_cpf in ["00000000000", "11111111111", "22222222222"]:
            return {"valid": False, "error": "Invalid CPF (known invalid)"}
        
        # Validate check digits
        for i in range(9, 11):
            value = sum((int(clean_cpf[num]) * ((i + 1) - num)) for num in range(0, i))
            digit = ((value * 10) % 11) % 10
            if digit != int(clean_cpf[i]):
                return {"valid": False, "error": "Invalid CPF (check digit failed)"}
        
        return {
            "valid": True,
            "country": "BR",
            "type": "CPF"
        }
    
    def _validate_south_african_id(self, id_number: str) -> Dict[str, Any]:
        """Validate South African ID Number."""
        if len(id_number) != 13:
            return {"valid": False, "error": "ID must be 13 digits"}
        
        # Apply Luhn algorithm
        if not self._luhn_check(id_number):
            return {"valid": False, "error": "Invalid ID (Luhn check failed)"}
        
        # Extract birth date
        year = int(id_number[0:2])
        month = int(id_number[2:4])
        day = int(id_number[4:6])
        
        # Validate date
        if month < 1 or month > 12:
            return {"valid": False, "error": "Invalid birth month"}
        
        if day < 1 or day > 31:
            return {"valid": False, "error": "Invalid birth day"}
        
        return {
            "valid": True,
            "country": "ZA",
            "type": "ID Number",
            "birth_year": 1900 + year if year > 50 else 2000 + year,
            "birth_month": month,
            "birth_day": day
        }
    
    def _validate_spanish_nie(self, nie: str) -> Dict[str, Any]:
        """Validate Spanish NIE."""
        if not re.match(r"^[XYZ]\d{7}[A-Z]$", nie.upper()):
            return {"valid": False, "error": "Invalid NIE format"}
        
        # Convert first letter to number
        letter_map = {"X": "0", "Y": "1", "Z": "2"}
        number = letter_map[nie[0]] + nie[1:8]
        
        # Calculate check letter
        check_letters = "TRWAGMYFPDXBNJZSQVHLCKE"
        expected_letter = check_letters[int(number) % 23]
        
        if nie[8].upper() != expected_letter:
            return {"valid": False, "error": "Invalid NIE check letter"}
        
        return {
            "valid": True,
            "country": "ES",
            "type": "NIE"
        }
    
    def validate_driver_license(self, license_number: str, state: str) -> Dict[str, Any]:
        """
        Validate US driver's license format.
        
        Args:
            license_number: License number
            state: Two-letter state code
            
        Returns:
            Validation result
        """
        state = state.upper()
        
        if state not in self.dl_formats:
            return {
                "valid": False,
                "error": f"Unknown state: {state}",
                "state": state
            }
        
        pattern = self.dl_formats[state]
        
        if re.match(pattern, license_number.upper()):
            return {
                "valid": True,
                "state": state,
                "type": "Driver's License",
                "real_id_compliant": self._check_real_id(state)  # After May 2025
            }
        
        return {
            "valid": False,
            "error": "Invalid format for state",
            "state": state,
            "expected_pattern": pattern
        }
    
    def _check_real_id(self, state: str) -> bool:
        """Check if state is REAL ID compliant (as of May 2025)."""
        # All states should be compliant by May 2025
        non_compliant = []  # Empty as of deadline
        return state not in non_compliant
    
    def validate_passport_mrz(self, mrz_lines: List[str]) -> Dict[str, Any]:
        """
        Validate passport Machine Readable Zone.
        
        Args:
            mrz_lines: List of MRZ lines
            
        Returns:
            Validation result
        """
        if len(mrz_lines) == 2 and len(mrz_lines[0]) == 44:
            mrz_type = "TD3"
        elif len(mrz_lines) == 2 and len(mrz_lines[0]) == 36:
            mrz_type = "TD2"
        elif len(mrz_lines) == 3 and len(mrz_lines[0]) == 30:
            mrz_type = "TD1"
        else:
            return {
                "valid": False,
                "error": "Invalid MRZ format"
            }
        
        # Concatenate lines
        mrz = "".join(mrz_lines)
        
        # Extract fields based on type
        if mrz_type == "TD3":  # Passport
            document_type = mrz[0:2]
            country = mrz[2:5]
            name = mrz[5:44].replace("<", " ").strip()
            document_number = mrz[44:53]
            document_check = mrz[53]
            nationality = mrz[54:57]
            birth_date = mrz[57:63]
            birth_check = mrz[63]
            sex = mrz[64]
            expiry_date = mrz[65:71]
            expiry_check = mrz[71]
            personal_number = mrz[72:86]
            personal_check = mrz[86]
            final_check = mrz[87]
            
            # Validate check digits
            checks_valid = (
                self._mrz_check_digit(document_number) == document_check and
                self._mrz_check_digit(birth_date) == birth_check and
                self._mrz_check_digit(expiry_date) == expiry_check
            )
            
            # Validate country code
            if country not in self.valid_country_codes:
                return {
                    "valid": False,
                    "error": f"Invalid country code: {country}"
                }
            
            return {
                "valid": checks_valid,
                "type": "Passport",
                "document_number": document_number,
                "country": country,
                "nationality": nationality,
                "name": name,
                "birth_date": self._parse_mrz_date(birth_date),
                "expiry_date": self._parse_mrz_date(expiry_date),
                "sex": sex
            }
        
        return {
            "valid": False,
            "error": f"MRZ type {mrz_type} not fully implemented"
        }
    
    def _mrz_check_digit(self, data: str) -> str:
        """Calculate MRZ check digit."""
        weights = [7, 3, 1]
        total = 0
        
        for i, char in enumerate(data):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char) - ord('A') + 10
            else:  # < or filler
                value = 0
            
            total += value * weights[i % 3]
        
        return str(total % 10)
    
    def _parse_mrz_date(self, date_str: str) -> str:
        """Parse MRZ date (YYMMDD) to ISO format."""
        if len(date_str) != 6:
            return "Invalid"
        
        year = int(date_str[0:2])
        # Assume 20xx for years 00-50, 19xx for 51-99
        year = 2000 + year if year <= 50 else 1900 + year
        
        month = date_str[2:4]
        day = date_str[4:6]
        
        return f"{year}-{month}-{day}"
    
    def validate_credit_card(self, card_number: str) -> Dict[str, Any]:
        """
        Validate credit card using Luhn algorithm.
        
        Args:
            card_number: Credit card number
            
        Returns:
            Validation result
        """
        # Remove spaces and hyphens
        clean_number = re.sub(r'[\s-]', '', card_number)
        
        if not clean_number.isdigit():
            return {
                "valid": False,
                "error": "Card number must contain only digits"
            }
        
        if len(clean_number) < 13 or len(clean_number) > 19:
            return {
                "valid": False,
                "error": "Invalid card number length"
            }
        
        # Apply Luhn algorithm
        if not self._luhn_check(clean_number):
            return {
                "valid": False,
                "error": "Invalid card number (Luhn check failed)"
            }
        
        # Identify card type
        card_type = self._identify_card_type(clean_number)
        
        return {
            "valid": True,
            "type": card_type,
            "masked": f"****{clean_number[-4:]}"
        }
    
    def _luhn_check(self, number: str) -> bool:
        """
        Perform Luhn algorithm check.
        
        Args:
            number: String of digits
            
        Returns:
            True if valid
        """
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        
        return checksum % 10 == 0
    
    def _identify_card_type(self, card_number: str) -> str:
        """Identify credit card type from number."""
        patterns = {
            "Visa": r"^4",
            "Mastercard": r"^5[1-5]|^2[2-7]",
            "American Express": r"^3[47]",
            "Discover": r"^6(?:011|5)",
            "JCB": r"^35",
            "Diners Club": r"^3(?:0[0-5]|[68])",
            "Maestro": r"^(?:5[06-8]|6)"
        }
        
        for card_type, pattern in patterns.items():
            if re.match(pattern, card_number):
                return card_type
        
        return "Unknown"
    
    def validate_document(self, document_data: str, document_type: str) -> Dict[str, Any]:
        """
        General document validation entry point.
        
        Args:
            document_data: Document number/data
            document_type: Type of document
            
        Returns:
            Validation result
        """
        validators = {
            "ssn": lambda d: self.validate_ssn(d),
            "driver_license": lambda d: self.validate_driver_license(d, "CA"),  # Default state
            "passport_mrz": lambda d: self.validate_passport_mrz(d.split('\n')),
            "credit_card": lambda d: self.validate_credit_card(d)
        }
        
        if document_type.lower() in validators:
            return validators[document_type.lower()](document_data)
        
        # Check national ID formats
        for format_key, format_info in self.national_id_formats.items():
            if re.match(format_info["format"], document_data):
                if "validator" in format_info:
                    return format_info["validator"](document_data)
                else:
                    return {
                        "valid": True,
                        "type": format_info["description"]
                    }
        
        return {
            "valid": False,
            "error": f"Unknown document type: {document_type}"
        }
    
    def validate_document_structure(self, document_type: str, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the structure and format of a document.
        
        Args:
            document_type: Type of document (driver_license, passport, etc.)
            document_data: Document data to validate
            
        Returns:
            Validation result with structure analysis
        """
        result = {
            "valid": False,
            "document_type": document_type,
            "errors": [],
            "warnings": [],
            "field_validation": {}
        }
        
        # Define required fields for each document type
        required_fields = {
            "driver_license": ["number", "name", "dob", "state", "expiry"],
            "passport": ["number", "name", "nationality", "dob", "expiry"],
            "ssn_card": ["number", "name"],
            "bank_statement": ["account_number", "name", "date", "balance"],
            "utility_bill": ["account_number", "name", "address", "date"],
            "insurance_card": ["policy_number", "name", "provider", "expiry"],
        }
        
        # Check if document type is supported
        if document_type not in required_fields:
            result["errors"].append(f"Unsupported document type: {document_type}")
            return result
        
        # Validate required fields
        missing_fields = []
        for field in required_fields[document_type]:
            if field not in document_data or not document_data[field]:
                missing_fields.append(field)
                result["field_validation"][field] = "missing"
            else:
                result["field_validation"][field] = "present"
        
        if missing_fields:
            result["errors"].append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Type-specific validation
        if document_type == "driver_license" and "number" in document_data:
            state = document_data.get("state", "CA")
            dl_validation = self.validate_driver_license(document_data["number"], state)
            result["field_validation"]["number"] = "valid" if dl_validation["valid"] else "invalid"
            if not dl_validation["valid"]:
                result["errors"].append(f"Invalid driver license format for {state}")
        
        elif document_type == "passport" and "number" in document_data:
            # Basic passport number validation
            passport_num = document_data["number"]
            if not re.match(r"^[A-Z0-9]{6,9}$", passport_num):
                result["errors"].append("Invalid passport number format")
                result["field_validation"]["number"] = "invalid"
        
        elif document_type == "ssn_card" and "number" in document_data:
            ssn_validation = self.validate_ssn(document_data["number"])
            result["field_validation"]["number"] = "valid" if ssn_validation["valid"] else "invalid"
            if not ssn_validation["valid"]:
                result["errors"].append("Invalid SSN format")
        
        # Check for suspicious patterns
        if document_data.get("name"):
            name = document_data["name"].lower()
            if any(fake in name for fake in ["test", "fake", "sample", "example", "mclovin"]):
                result["warnings"].append("Suspicious name detected")
                result["field_validation"]["name"] = "suspicious"
        
        # Date validation
        date_fields = ["dob", "expiry", "date"]
        for field in date_fields:
            if field in document_data:
                try:
                    from datetime import datetime
                    # Try to parse date
                    date_str = document_data[field]
                    if isinstance(date_str, str):
                        datetime.strptime(date_str, "%Y-%m-%d")
                        result["field_validation"][field] = "valid"
                except:
                    result["warnings"].append(f"Invalid date format for {field}")
                    result["field_validation"][field] = "invalid"
        
        # Determine overall validity
        result["valid"] = len(result["errors"]) == 0
        
        return result
    
    def get_supported_documents(self) -> Dict[str, List[str]]:
        """Get list of supported document types."""
        return {
            "social_security": list(self.ssn_formats.keys()),
            "driver_licenses": list(self.dl_formats.keys()),
            "passport_types": list(self.mrz_formats.keys()),
            "national_ids": list(self.national_id_formats.keys()),
            "other": ["credit_card"]
        }