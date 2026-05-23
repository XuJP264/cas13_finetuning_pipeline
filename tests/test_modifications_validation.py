#!/usr/bin/env python
"""Test script to validate all the modifications made for H100/Merlin compatibility"""
import subprocess
import sys
from pathlib import Path

def test_esmfold_script_help():
    """Test that 19_esmfold_batch_oracle.py has all required arguments including --chunk-size"""
    print("Testing ESMFold script --help output...")
    result = subprocess.run(
        [sys.executable, "scripts/19_esmfold_batch_oracle.py", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "ESMFold script failed to run"
    help_text = result.stdout
    required_args = ["--input", "--output", "--model", "--device", "--max-length", "--chunk-size"]
    for arg in required_args:
        assert arg in help_text, f"Missing argument {arg} in ESMFold script help"
    print("✓ All required arguments found in ESMFold script")

def test_progen3_default_disabled():
    """Test that ProGen3Oracle is disabled by default"""
    print("\nTesting ProGen3 default disabled status...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from cas13_rl.oracle_progen3 import ProGen3Oracle
    oracle = ProGen3Oracle()
    assert oracle.mode == "disabled", "ProGen3Oracle default mode should be 'disabled'"
    
    # Test that disabled mode returns correct error
    result = oracle.score_one("ABC")
    assert result["valid"] is False, "Disabled ProGen3 should return invalid"
    assert "ProGen3 is disabled" in result["error"], "Should include disabled error message"
    assert result["backend"] == "disabled", "Backend should be marked as disabled"
    print("✓ ProGen3 correctly disabled by default")

def test_esmfold_oracle_config_fields():
    """Test that ESMFoldOracle has all required configuration fields"""
    print("\nTesting ESMFoldOracle configuration fields...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from cas13_rl.oracle_esmfold import ESMFoldOracle
    oracle = ESMFoldOracle()
    
    # Check all required fields exist with correct defaults
    required_fields = {
        "model": "facebook/esmfold_v1",
        "python": ".venv_oracle/bin/python",
        "script": "scripts/19_esmfold_batch_oracle.py",
        "device": "cuda:0",
        "max_length": 1500,
        "chunk_size": 64,
        "cuda_visible_devices": "1"
    }
    
    for field, expected_value in required_fields.items():
        assert hasattr(oracle, field), f"Missing field {field} in ESMFoldOracle"
        assert getattr(oracle, field) == expected_value, f"Field {field} has wrong default value"
    print("✓ All ESMFoldOracle configuration fields present with correct defaults")

def test_config_files_exist():
    """Test that all required config files were created"""
    print("\nTesting all required config files exist...")
    config_dir = Path(__file__).parent.parent / "configs"
    required_configs = [
        "sft_cas13_11000_2epoch.yaml",
        "rl_cas13_sft_best_mock_smoke.yaml",
        "rl_cas13_sft_best_esmfold_smoke.yaml",
        "rl_cas13_sft_best_esmfold.yaml"
    ]
    
    for config in required_configs:
        config_path = config_dir / config
        assert config_path.exists(), f"Missing config file {config}"
        print(f"  ✓ {config} exists")

def test_extraction_script_exists():
    """Test that the data extraction script exists"""
    print("\nTesting data extraction script...")
    script_path = Path(__file__).parent.parent / "scripts/extract_cas13_11000_no_dedup.py"
    assert script_path.exists(), "Missing extraction script extract_cas13_11000_no_dedup.py"
    print("✓ Extraction script exists")

if __name__ == "__main__":
    print("Running validation tests for H100/Merlin modifications...\n")
    try:
        test_esmfold_script_help()
        test_progen3_default_disabled()
        test_esmfold_oracle_config_fields()
        test_config_files_exist()
        test_extraction_script_exists()
        
        print("\n✅ All validation tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)