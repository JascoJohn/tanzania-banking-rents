"""Excel companion model (D104): a chapter-organised workbook that reproduces the Python pipeline with live formulas.

``python -m tzbank run excel_model`` (or ``scripts/20_excel_model.py``) builds
``outputs/valuation/Tanzania_Banking_Rents_Model.xlsx``, recalculates it in Excel (or LibreOffice) and compares every
formula-driven key output with the Python tables (``outputs/valuation/excel_model_check.csv``). Python remains the
source of truth; the workbook reproduces it.
"""
