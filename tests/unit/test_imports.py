"""모듈 import 테스트"""

import pytest


def test_import_weighted_mtp():
    """weighted_mtp 패키지 import"""
    import weighted_mtp

    assert weighted_mtp.__version__ == "0.2.0"


def test_import_submodules():
    """하위 모듈 import"""
    import weighted_mtp.cli
    import weighted_mtp.core
    import weighted_mtp.data
    import weighted_mtp.models
    import weighted_mtp.value_weighting
    import weighted_mtp.pipelines
    import weighted_mtp.runtime
    import weighted_mtp.utils
