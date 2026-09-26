# Test documents

| File | Source | What it tests |
|---|---|---|
| `01_web_hosting_agreement.txt` | CUAD (real contract, SEC EDGAR filing) | Normal Q&A: renewal, termination, liability cap |
| `02_non_compete_agreement.txt` | CUAD | Tiny document (~500 words): most questions should be refused |
| `03_franchise_agreement_long.txt` | CUAD | Long document (~23k words): upload time, many clause types |
| `04_residential_lease.docx` | Synthetic, written for testing | DOCX parsing; consumer-style questions that invite advice |
| `05_mutual_nda_2_pages.pdf` | Synthetic, written for testing | PDF parsing and page-numbered citations |
| `06_injection_test.txt` | Synthetic | Prompt injection planted inside a clause |
| `07_scanned_page.png` | Synthetic | Unsupported input: image/OCR is out of scope, expect a clean 400 |

CUAD contracts: Contract Understanding Atticus Dataset, CC BY 4.0 (The Atticus Project).
The synthetic documents are fictional and are not legal templates.
