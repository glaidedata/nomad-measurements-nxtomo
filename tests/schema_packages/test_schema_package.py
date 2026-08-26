from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nomad.datamodel import EntryArchive
from nomad.datamodel.datamodel import EntryMetadata
from readers_ientrance.rcp_reader import RcpData, RecipePoint
from readers_ientrance.txm_reader import TxmData
from readers_ientrance.txrm_reader import TxrmData

from nomad_measurements_nxtomo.schema_packages.schema_package import (
    ELNZeissRecipe,
    ELNZeissTXM,
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


# Bypass HDF5 physical file writing during the unit test (Removed 'self' from lambda)
@patch(
    'nomad.datamodel.hdf5.HDF5Dataset._normalize_impl',
    side_effect=lambda value, **kwargs: value,
)
@patch('nomad_measurements_nxtomo.schema_packages.schema_package.extract_preview_image')
@patch('nomad_measurements_nxtomo.schema_packages.schema_package.read_txrm')
def test_eln_zeiss_txrm_normalization(
    mock_read_txrm, mock_extract_image, mock_hdf5_norm, mock_archive
):
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

    # Mock the numpy array returned by the image extractor
    mock_extract_image.return_value = np.zeros((1010, 1010), dtype=np.uint16)

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

    # Assert preview image mapping
    assert result.preview_image is not None
    assert result.preview_image.shape == (1010, 1010)


# Bypass HDF5 physical file writing during the unit test (Removed 'self' from lambda)
@patch(
    'nomad.datamodel.hdf5.HDF5Dataset._normalize_impl',
    side_effect=lambda value, **kwargs: value,
)
@patch('nomad_measurements_nxtomo.schema_packages.schema_package.extract_preview_image')
@patch('nomad_measurements_nxtomo.schema_packages.schema_package.read_txm')
def test_eln_zeiss_txm_normalization(
    mock_read_txm, mock_extract_image, mock_hdf5_norm, mock_archive
):
    """Test the metadata mapping logic for the TXM schema."""
    # Define constants to avoid PLR2004 magic value linting errors
    expected_mag = 40.0
    expected_voltage = 80.0
    expected_results_count = 1
    expected_slices = 1000
    expected_image_data = 250

    # 1. Prepare Mock Data returned by the reader
    mock_data = TxmData()
    mock_data.metadata = {
        'Version': '16.2.1',
        'Total_3D_Slices_or_Blocks': expected_slices,
    }
    # Moved hardware setups to recon_settings for TXM
    mock_data.recon_settings = {
        'LensMagnification': {'int32': 1100924689, 'float32': expected_mag},
        'SourceVoltage': {'int32': 1116471296, 'float32': expected_voltage},
        'VoxelSize': {'float32': 2.5},
    }
    mock_data.image_data_summary = {
        'ImageData1': expected_image_data,
    }

    mock_read_txm.return_value = mock_data

    # Mock the numpy array returned by the image extractor
    # We add a non-zero value so it passes the empty-slice (min != max) safety check
    dummy_image = np.zeros((989, 1010), dtype=np.uint16)
    dummy_image[0, 0] = 65535  # Add a single bright pixel
    mock_extract_image.return_value = dummy_image

    # 2. Initialize the ELN Entry
    entry = ELNZeissTXM()
    entry.data_file = 'test.txm'

    # 3. Run Normalization
    entry.normalize(mock_archive, logger=None)

    # 4. Assert Data Mapping
    assert entry.software_version == '16.2.1'
    assert entry.instrument_setup.objective_magnification == expected_mag
    assert entry.instrument_setup.source_voltage.magnitude == expected_voltage
    assert 'VoxelSize' in entry.raw_recon_settings

    assert len(entry.results) == expected_results_count
    result = entry.results[0]
    assert result.total_slices == expected_slices
    assert result.image_data_catalog['ImageData1'] == expected_image_data

    # Assert preview image mapping
    assert result.preview_image is not None
    assert result.preview_image.shape == (989, 1010)


@patch('nomad_measurements_nxtomo.schema_packages.schema_package.read_rcp')
def test_recipe_normalization_fails_on_reader_extraction_error(
    mock_read_rcp, mock_archive
):
    """Reader failures stop RCP mapping and surface as normalization errors."""
    mock_read_rcp.return_value = RcpData(
        metadata={
            'extraction_error': 'invalid recipe container',
            'RecipeName': 'must not be mapped',
            'NoOfTomoDataSets': {'int32': 2},
        }
    )
    entry = ELNZeissRecipe(data_file='invalid.rcp')

    with pytest.raises(
        ValueError,
        match='RCP reader extraction failed: invalid recipe container',
    ):
        entry.normalize(mock_archive, logger=None)

    assert entry.raw_metadata is None
    assert entry.recipe_name is None
    assert entry.number_of_datasets is None
    assert not entry.recipe_points


@patch('nomad_measurements_nxtomo.schema_packages.schema_package.extract_preview_image')
@patch('nomad_measurements_nxtomo.schema_packages.schema_package.read_txrm')
def test_txrm_normalization_fails_on_reader_extraction_error(
    mock_read_txrm, mock_extract_image, mock_archive
):
    """Reader failures stop TXRM mapping before setup and result creation."""
    mock_read_txrm.return_value = TxrmData(
        metadata={
            'extraction_error': 'invalid acquisition container',
            'Version': 'must not be mapped',
            'Total_Projections': 2401,
        },
        acquisition_settings={'ObjectiveMag': {'float32': 20.0}},
    )
    entry = ELNZeissTXRM(data_file='invalid.txrm')

    with pytest.raises(
        ValueError,
        match='TXRM reader extraction failed: invalid acquisition container',
    ):
        entry.normalize(mock_archive, logger=None)

    assert entry.raw_metadata is None
    assert entry.software_version is None
    assert entry.instrument_setup is None
    assert entry.acquisition_setup is None
    assert not entry.results
    mock_extract_image.assert_not_called()


@patch('nomad_measurements_nxtomo.schema_packages.schema_package.extract_preview_image')
@patch('nomad_measurements_nxtomo.schema_packages.schema_package.read_txm')
def test_txm_normalization_fails_on_reader_extraction_error(
    mock_read_txm, mock_extract_image, mock_archive
):
    """Reader failures stop TXM mapping before setup and result creation."""
    mock_read_txm.return_value = TxmData(
        metadata={
            'extraction_error': 'invalid reconstruction container',
            'Version': 'must not be mapped',
            'Total_3D_Slices_or_Blocks': 1000,
        },
        recon_settings={'LensMagnification': {'float32': 40.0}},
    )
    entry = ELNZeissTXM(data_file='invalid.txm')

    with pytest.raises(
        ValueError,
        match='TXM reader extraction failed: invalid reconstruction container',
    ):
        entry.normalize(mock_archive, logger=None)

    assert entry.raw_metadata is None
    assert entry.software_version is None
    assert entry.raw_recon_settings is None
    assert entry.instrument_setup is None
    assert not entry.results
    mock_extract_image.assert_not_called()


@patch('nomad_measurements_nxtomo.schema_packages.schema_package.extract_preview_image')
def test_txrm_normalization_fails_for_actual_invalid_file(
    mock_extract_image, mock_archive, tmp_path
):
    """Structural reader validation rejects an actual non-OLE file on disk."""
    invalid_file = tmp_path / 'invalid.txrm'
    invalid_file.write_bytes(b'not an OLE2 container')
    mock_archive.m_context.upload_files.raw_file_object.return_value.os_path = str(
        invalid_file
    )
    entry = ELNZeissTXRM(data_file='invalid.txrm')

    with pytest.raises(
        ValueError,
        match='TXRM reader extraction failed: Not a valid OLE2 file:',
    ):
        entry.normalize(mock_archive, logger=None)

    assert entry.raw_metadata is None
    assert entry.software_version is None
    assert entry.instrument_setup is None
    assert entry.acquisition_setup is None
    assert not entry.results
    mock_extract_image.assert_not_called()
