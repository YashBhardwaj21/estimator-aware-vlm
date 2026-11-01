"""
Script to scan the repository for hardcoded empirical values and potential integrity violations.
Searches for direct assignments to empirical metrics, placeholder dictionaries,
formulaic offsets, and mock model responses.
Outputs to results/publication_campaign/run_20261101_canonical/audit/source_integrity_audit.md.
"""

import os
import re
import ast

TARGET_DIRS = ["scripts", "src", "mapping", "native", "configs"]
SUSPICIOUS_VARS = [
    r"\bate\b\s*=",
    r"\bate_rmse\b\s*=",
    r"\brpe\b\s*=",
    r"\brpe_trans\b\s*=",
    r"\brpe_rot\b\s*=",
    r"\bbaseline_ate\b\s*=",
    r"\bintervention_ate\b\s*=",
    r"\bimprovement\b\s*=",
    r"\bdelta_ate\b\s*=",
    r"\blatency\b\s*=",
    r"\baccuracy\b\s*=",
    r"\bprecision\b\s*=",
    r"\brecall\b\s*=",
    r"\bf1\b\s*=",
    r"\bdisplacement\b\s*=",
    r"\bresidual\b\s*=",
    r"\binlier_count\b\s*=",
    r"\btoken_count\b\s*=",
    r"\bpredicted_label\b\s*="
]

FORMULAIC_PATTERNS = [
    r"base_ate\s*[-+]\s*\d+\.\d+",
    r"baseline_ate\s*[-+]\s*\d+\.\d+",
    r"residual_lookup\s*=",
    r"ate_lookup\s*=",
    r"mock_vlm",
    r"mock_response",
    r"round\(base_ate\s*[-+]"
]

def scan_files():
    findings = []
    
    for tdir in TARGET_DIRS:
        if not os.path.exists(tdir):
            continue
        for root, _, files in os.walk(tdir):
            for file in files:
                if not file.endswith((".py", ".cpp", ".h", ".sh")):
                    continue
                fpath = os.path.join(root, file)
                relpath = os.path.relpath(fpath)
                
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    
                for idx, line in enumerate(lines, 1):
                    strip_line = line.strip()
                    if strip_line.startswith("#") or strip_line.startswith("//"):
                        continue
                        
                    # Check for formulaic patterns
                    for pat in FORMULAIC_PATTERNS:
                        if re.search(pat, line):
                            findings.append({
                                "file": relpath,
                                "line": idx,
                                "severity": "CRITICAL",
                                "code": strip_line,
                                "type": "Formulaic / Synthetic Calculation Pattern",
                                "pattern": pat
                            })
                            
                    # Check for suspicious var assignments
                    for var_pat in SUSPICIOUS_VARS:
                        if re.search(var_pat, line, re.IGNORECASE):
                            # Filter out legitimate assignments like math expressions or parser extraction
                            if any(safe in line for safe in ["get(", "parse", "float(", "np.", "math.", "compute_", "calculate_", "eval", "item()", "load"]):
                                continue
                            # If assigning a literal float or int
                            if re.search(r"=\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(?:#.*)?$", strip_line):
                                findings.append({
                                    "file": relpath,
                                    "line": idx,
                                    "severity": "HIGH",
                                    "code": strip_line,
                                    "type": "Literal Empirical Assignment",
                                    "pattern": var_pat
                                })
    return findings

def generate_report(findings, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SOURCE CODE EMPIRICAL INTEGRITY AUDIT REPORT\n\n")
        f.write("**Audit ID**: AUD-SRC-ZERO-HARDCODING-20261101\n")
        f.write("**Standard**: Zero-Hardcoding, Zero-Simulation, Absolute Research Integrity\n")
        f.write(f"**Total Findings**: {len(findings)}\n\n")
        f.write("## 1. Summary of Detected Integrity Violations\n\n")
        
        crit = sum(1 for x in findings if x["severity"] == "CRITICAL")
        high = sum(1 for x in findings if x["severity"] == "HIGH")
        med = sum(1 for x in findings if x["severity"] == "MEDIUM")
        
        f.write(f"- **CRITICAL (Formulaic calculations, lookups, synthetic offsets)**: {crit}\n")
        f.write(f"- **HIGH (Direct literal numeric assignments to metric variables)**: {high}\n")
        f.write(f"- **MEDIUM (Suspicious defaults or fallback logic)**: {med}\n\n")
        
        f.write("## 2. Granular Code Finding Inventory\n\n")
        f.write("| File | Line | Severity | Category | Code Extract | Required Correction | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for fnd in findings:
            code_snip = fnd["code"].replace("|", "\\|")
            if len(code_snip) > 65:
                code_snip = code_snip[:62] + "..."
            corr = "Replace with raw trajectory calculation" if "ate" in fnd["code"] else "Compute from raw artifacts"
            f.write(f"| `{fnd['file']}` | {fnd['line']} | **{fnd['severity']}** | {fnd['type']} | `{code_snip}` | {corr} | `FLAGGED_FOR_REPAIR` |\n")
            
        f.write("\n## 3. Mandatory Remediation Directives\n\n")
        f.write("1. All formulaic arithmetic (such as `round(base_ate - 0.088, 4)`) must be excised.\n")
        f.write("2. All lookup dictionaries for residuals or ATE must be removed from production experiment paths.\n")
        f.write("3. All metric variables must be populated solely by parsing raw trajectory files against reference poses using Umeyama SE(3) alignment.\n")
        f.write("4. Automated anti-fabrication assertions must be installed in pipeline runners to fail-closed on unexecuted samples.\n")

if __name__ == "__main__":
    findings = scan_files()
    out = "results/publication_campaign/run_20261101_canonical/audit/source_integrity_audit.md"
    generate_report(findings, out)
    print(f"Audit completed: {len(findings)} findings written to {out}")

