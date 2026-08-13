from unittest.mock import MagicMock, patch

import pytest
from nomad.datamodel import EntryArchive
from nomad.datamodel.datamodel import EntryMetadata
from readers_ientrance.rcp_reader import RcpData, RecipePoint
from readers_ientrance.txrm_reader import TxrmData

from nomad_measurements_nxtomo.schema_packages.schema_package import (
    ELNZeissRecipe,
    ELNZeissTXRM,
)


@pytest.fixture
def mock_archive():
    """Creates a mock NOMAD EntryArchive with a simulated file context and metadata."""
    archive = EntryArchive()
    archive.m_context = MagicMock()
    # Simulate NOMAD finding the raw uploaded file path
    archive.m_context.upload_files.raw_file_object.return_value.os_path = (
        'mocked_path.file'
    )

    # Use real EntryMetadata to satisfy NOMAD and prevent SyntaxWarnings
    archive.metadata = EntryMetadata(entry_name='test_entry.archive.json')
    return archive


@patch('nomad_measurements_nxtomo.schema_packages.schema_package.read_rcp')
def test_eln_zeiss_recipe_normalization(mock_read_rcp, mock_archive):
    """Test the metadata mapping logic for the RCP schema."""
    # Define constants to avoid PLR2004 magic value linting errors
    expected_datasets = 2
    expected_points_count = 1
    expected_voltage = 70.0
    expected_exp_time = 1.5
    expected_total_images = 1200

    # 1. Prepare Mock Data returned by the reader
    mock_data = RcpData()
    mock_data.metadata = {
        'RecipeName': 'Test_Battery_Scan',
        'NoOfTomoDataSets': {'int32': expected_datasets, 'float32': 2.0},
    }

    rp = RecipePoint()
    rp.metadata = {'PointName': 'Scan1'}
    rp.acquisition_settings.metadata = {
        'SrcVoltage': {'int32': 1116471296, 'float32': expected_voltage},
        'ExpTime': {'int32': 1065353216, 'float32': expected_exp_time},
        'TotalImages': {'int32': expected_total_images, 'float32': 0.0},
    }
    mock_data.recipe_points = {'RecipePoint0': rp}

    mock_read_rcp.return_value = mock_data

    # 2. Initialize the ELN Entry
    entry = ELNZeissRecipe()
    entry.data_file = 'test.rcp'

    # 3. Run Normalization
    entry.normalize(mock_archive, logger=None)

    # 4. Assert Data Mapping
    assert entry.recipe_name == 'Test_Battery_Scan'
    assert entry.number_of_datasets == expected_datasets
    assert len(entry.recipe_points) == expected_points_count

    mapped_point = entry.recipe_points[0]
    assert mapped_point.point_name == 'Scan1'

    # Append .magnitude to extract the raw float from NOMAD Quantities with units
    assert mapped_point.instrument_setup.source_voltage.magnitude == expected_voltage
    assert mapped_point.acquisition_setup.exposure_time.magnitude == expected_exp_time

    # total_images has no unit in the schema, so it remains a raw integer
    assert mapped_point.acquisition_setup.total_images == expected_total_images


@patch('nomad_measurements_nxtomo.schema_packages.schema_package.read_txrm')
def test_eln_zeiss_txrm_normalization(mock_read_txrm, mock_archive):
    """Test the metadata mapping logic for the TXRM schema."""
    # Define constants to avoid PLR2004 magic value linting errors
    expected_mag = 20.0
    expected_results_count = 1
    expected_projections = 2401
    expected_image_data = 100

    # 1. Prepare Mock Data returned by the reader
    mock_data = TxrmData()
    mock_data.metadata = {
        'Version': '16.2.1',
        'Total_Projections': expected_projections,
    }
    mock_data.acquisition_settings = {
        'ObjectiveMag': {'int32': 1100924689, 'float32': expected_mag},
    }
    mock_data.image_data_summary = {
        'ImageData1': expected_image_data,
        'ImageData2': 50,
    }

    mock_read_txrm.return_value = mock_data

    # 2. Initialize the ELN Entry
    entry = ELNZeissTXRM()
    entry.data_file = 'test.txrm'

    # 3. Run Normalization
    entry.normalize(mock_archive, logger=None)

    # 4. Assert Data Mapping
    assert entry.software_version == '16.2.1'
    # Objective magnification has no unit, so no .magnitude is needed
    assert entry.instrument_setup.objective_magnification == expected_mag

    assert len(entry.results) == expected_results_count
    result = entry.results[0]
    assert result.total_projections == expected_projections
    assert result.image_data_catalog['ImageData1'] == expected_image_data
