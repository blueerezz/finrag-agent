from finrag.ingestion.chunk import chunk_document

INFO = {"doc_id": "GOOGL_10K_2024-12-31", "ticker": "GOOGL", "form": "10-K",
        "period_end": "2024-12-31", "fiscal_year": 2024}


def _doc() -> str:
    rows = "\n".join(f"| Line item {i} | ${i * 1000:,} | ${i * 900:,} |" for i in range(1, 200))
    return (
        "Item 8. Financial Statements and Supplementary Data\n\n"
        "Alphabet Inc.\nCONSOLIDATED STATEMENTS OF INCOME\n(in millions)\n\n"
        "| | Year Ended December 31, | |\n| --- | --- | --- |\n| | 2023 | 2024 |\n" + rows + "\n"
    )


def test_long_table_splits_and_every_piece_keeps_its_header():
    tables = [c for c in chunk_document(_doc(), INFO) if c.kind == "table"]
    assert len(tables) > 1
    for c in tables:
        assert "| | 2023 | 2024 |" in c.text
        assert "(in millions)" in c.text
        assert c.section.startswith("Item 8")
        assert len(c.embed_text) <= 1500


def test_no_data_row_is_lost_or_duplicated():
    tables = [c for c in chunk_document(_doc(), INFO) if c.kind == "table"]
    joined = "\n".join(c.text for c in tables)
    for i in range(1, 200):
        assert joined.count(f"| Line item {i} |") == 1
def test_item_heading_inside_table_row_is_detected():
    doc = (
        "| ITEM 7. | MANAGEMENT’S DISCUSSION AND ANALYSIS |\n| --- | --- |\n\n"
        + "Revenue grew because of strong demand. " * 5 + "\n"
    )
    chunks = chunk_document(doc, INFO)
    assert chunks and all(c.section.startswith("Item 7.") for c in chunks)


def test_table_of_contents_row_is_not_a_heading():
    doc = "| ITEM 7. | Management's Discussion | 34 |\n| --- | --- | --- |\n"
    assert all(not c.section.startswith("Item 7.") for c in chunk_document(doc, INFO))

def test_financials_filed_under_item15_are_relabeled_to_item8():
    doc = (
        "Item 8. Financial Statements and Supplementary Data\n"
        "The information required by this Item is set forth in our "
        "Consolidated Financial Statements included in this Annual Report.\n\n"
        "Item 9. Changes in and Disagreements with Accountants\nNone.\n\n"
        "Item 15. Exhibits, Financial Statement Schedule\n\n"
        "| Report of Independent Registered Public Accounting Firm | 39 |\n"
        "| --- | --- |\n"
        "| Consolidated Statements of Income | 41 |\n\n"
        "NVIDIA CORPORATION AND SUBSIDIARIES\nCONSOLIDATED BALANCE SHEETS\n"
        "(In millions)\n\n"
        "| Assets | 2019 |\n| --- | --- |\n| Cash | 7,000 |\n\n"
     "NOTE 1. SUMMARY OF SIGNIFICANT ACCOUNTING POLICIES\n"
        "We recognize revenue when the customer obtains control of the product.\n\n"
        "| Signature | Title |\n"
        "| --- | --- |\n"
        "| /s/ John Doe | CEO |\n\n"
        "Index to Exhibits\n\n"
        "Exhibit 10.1 - Form of Indemnification Agreement\n"
    )
    chunks = chunk_document(doc, INFO)
    fin = [c for c in chunks if "CONSOLIDATED BALANCE SHEETS" in c.text
           or "Consolidated Statements of Income" in c.text
           or "NOTE 1." in c.text]
    assert fin, "expected financial chunks"
    assert all(c.section.startswith("Item 8") for c in fin)
    exhibits = [c for c in chunks if "Indemnification" in c.text]
    assert exhibits and all(c.section.startswith("Item 15") for c in exhibits)


def test_pep_style_financials_inside_item7_are_relabeled():
    # Realistic PEP: MD&A prose, then statement heading, notes, signature
    # table, then exhibit index. Signature table forces a chunk boundary
    # so the notes land separately from the exhibit index.
    doc = (
        "Item 7. Management's Discussion and Analysis\n\n"
        "Our consolidated net revenue grew 12% year over year.\n\n"
        "The following table presents our Consolidated Statements of Income "
        "as a percentage of revenue.\n\n"
        "| Net revenue | 100% |\n| --- | --- |\n| Cost of sales | 45% |\n\n"
        + "MD&A paragraph continues with more narrative. " * 20 + "\n\n"
        "CONSOLIDATED STATEMENTS OF INCOME\n"
        "(in millions)\n\n"
        "| | 2021 | 2020 |\n| --- | --- | --- |\n"
        "| Net revenue | 79,474 | 70,372 |\n\n"
        "NOTE 1. BASIS OF PRESENTATION\n"
        "The accompanying consolidated financial statements...\n\n"
        "| Signature | Title |\n"
        "| --- | --- |\n"
        "| /s/ Jane Doe | CFO |\n\n"
        "Index to Exhibits\n\n"
        "Exhibit 10.1 - Form of Agreement\n"
    )
    chunks = chunk_document(doc, INFO)

    fin = [c for c in chunks if "CONSOLIDATED STATEMENTS OF INCOME" in c.text
           or "NOTE 1." in c.text]
    assert fin
    assert all(c.section.startswith("Item 8") for c in fin)

    mda = [c for c in chunks if "as a percentage of revenue" in c.text]
    assert mda and all(c.section.startswith("Item 7.") for c in mda)
