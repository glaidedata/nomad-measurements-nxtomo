from typing import TYPE_CHECKING

import numpy as np
from nomad.datamodel.data import JSON, ArchiveSection, EntryData
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.datamodel.metainfo.annotations import ELNComponentEnum, H5WebAnnotation
from nomad.datamodel.metainfo.basesections import Measurement, MeasurementResult
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

# Import the readers from your extended readers package
from readers_ientrance import read_rcp, read_txm, read_txrm
from readers_ientrance.txrm_reader import extract_preview_image

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

m_package = SchemaPackage()


# ==========================================
# 1. SHARED NXTOMO SETUP SECTIONS
# ==========================================
class NXtomoInstrumentSetup(ArchiveSection):
    """Details about the X-ray microscope hardware."""

    source_voltage = Quantity(
        type=np.float64, unit='kV', description='X-ray source voltage.'
    )
    objective_magnification = Quantity(
        type=np.float64, description='Microscope objective magnification.'
    )


class NXtomoAcquisitionSetup(ArchiveSection):
    """Parameters governing the tomographic scan execution."""

    exposure_time = Quantity(
        type=np.float64, unit='s', description='Exposure time per projection.'
    )
    total_images = Quantity(
        type=np.int32, description='Total number of radiographic projections.'
    )
    acquisition_mode = Quantity(type=str, description='Acquisition mode or scan type.')


class NXtomoRecipePoint(ArchiveSection):
    """A sub-section to store individual scan/warmup steps from a Recipe."""

    point_name = Quantity(
        type=str, description='Name of the recipe point (e.g., WarmupA, Scan1).'
    )
    instrument_setup = SubSection(section_def=NXtomoInstrumentSetup)
    acquisition_setup = SubSection(section_def=NXtomoAcquisitionSetup)
    raw_acquisition_settings = Quantity(
        type=JSON, description='Raw acquisition metadata stream.'
    )
    raw_recon_settings = Quantity(
        type=JSON, description='Raw reconstruction metadata stream.'
    )


# ==========================================
# 2. SHARED NXTOMO RESULTS
# ==========================================
class NXtomoResult(MeasurementResult):
    """Holds analytical data or lazy catalogs of the projections."""

    m_def = Section(a_h5web=H5WebAnnotation(signal='preview_image'))

    preview_image = Quantity(
        type=HDF5Dataset,
        description='A 2D projection extracted from the file for preview purposes.',
        a_eln=dict(overview=True),
    )
    total_projections = Quantity(
        type=np.int32,
        description='Actual total number of projections found in the file.',
    )
    total_slices = Quantity(
        type=np.int32,
        description='Total number of 3D volume slices or blocks found in the TXM file.',
    )
    image_data_catalog = Quantity(
        type=JSON,
        description='A summary of the projection image directories and their counts.',
    )
    temperature_info = Quantity(
        type=JSON, description='Temperature sensor records during acquisition.'
    )
    hardware_stability = Quantity(
        type=JSON, description='Hardware stability and drift metrics.'
    )


# ==========================================
# 3. BASE NXTOMO ENTRY
# ==========================================
class BaseNXtomoMeasurement(Measurement):
    """Base class containing shared attributes for NXtomo entries."""

    data_file = Quantity(
        type=str,
        a_eln=dict(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor='RawFileAdaptor'),
        description='The raw NXtomo data file (.txrm or .rcp).',
    )
    instrument_model = Quantity(
        type=str,
        default='ZEISS Xradia Versa 610',
        description='The model of the X-ray microscope.',
        a_eln=dict(component=ELNComponentEnum.StringEditQuantity),
    )
    software_version = Quantity(
        type=str,
        description='Software used to record the measurements (e.g., Scout-and-Scan).',
        a_eln=dict(component=ELNComponentEnum.StringEditQuantity),
    )
    raw_metadata = Quantity(
        type=JSON,
        description='Global file-level metadata.',
    )


# ==========================================
# 4. ELN ENTRY: ZEISS RECIPE (.rcp)
# ==========================================
class ELNZeissRecipe(BaseNXtomoMeasurement, EntryData):
    m_def = Section(
        label='ZEISS NXtomo Recipe',
        a_eln=dict(lane_width='600px'),
        a_template=dict(measurement_identifiers=dict()),
    )

    recipe_name = Quantity(type=str, description='Name of the tomographic recipe.')
    number_of_datasets = Quantity(
        type=np.int32, description='Expected number of data sets in this sequence.'
    )

    recipe_points = SubSection(section_def=NXtomoRecipePoint, repeats=True)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if not self.data_file:
            super().normalize(archive, logger)
            return

        try:
            file_path = archive.m_context.upload_files.raw_file_object(
                self.data_file
            ).os_path
            rcp_data = read_rcp(file_path)

            if 'extraction_error' in rcp_data.metadata:
                logger.warning(
                    f'RCP Reader Warning: {rcp_data.metadata["extraction_error"]}'
                )

            # Top-level Metadata
            self.raw_metadata = rcp_data.metadata
            self.recipe_name = str(rcp_data.metadata.get('RecipeName', 'Unknown'))
            num_ds = rcp_data.metadata.get('NoOfTomoDataSets', {})
            if isinstance(num_ds, dict) and 'int32' in num_ds:
                self.number_of_datasets = num_ds['int32']

            self.recipe_points = []

            # Map sequential recipe points
            for pt_name, pt_data in rcp_data.recipe_points.items():
                point_sec = NXtomoRecipePoint()
                point_sec.point_name = str(pt_data.metadata.get('PointName', pt_name))

                acq_settings = pt_data.acquisition_settings.metadata
                point_sec.raw_acquisition_settings = acq_settings
                point_sec.raw_recon_settings = pt_data.recon_settings.metadata

                # Setup Instrument & Acquisition sub-sections
                inst_setup = NXtomoInstrumentSetup()
                acq_setup = NXtomoAcquisitionSetup()

                # Safely extract dual-decoded numeric values
                src_voltage = acq_settings.get('SrcVoltage', {})
                if isinstance(src_voltage, dict) and 'float32' in src_voltage:
                    inst_setup.source_voltage = src_voltage['float32']

                obj_mag = acq_settings.get('ObjectiveMag', {})
                if isinstance(obj_mag, dict) and 'float32' in obj_mag:
                    inst_setup.objective_magnification = obj_mag['float32']

                exp_time = acq_settings.get('ExpTime', {})
                if isinstance(exp_time, dict) and 'float32' in exp_time:
                    acq_setup.exposure_time = exp_time['float32']

                tot_img = acq_settings.get('TotalImages', {})
                if isinstance(tot_img, dict) and 'int32' in tot_img:
                    acq_setup.total_images = tot_img['int32']

                point_sec.instrument_setup = inst_setup
                point_sec.acquisition_setup = acq_setup
                self.recipe_points.append(point_sec)

        except Exception as e:
            if logger:
                logger.error(f'Error parsing ZEISS RCP file: {e}')
            raise e

        super().normalize(archive, logger)


# ==========================================
# 5. ELN ENTRY: ZEISS RAW RECORD (.txrm)
# ==========================================
class ELNZeissTXRM(BaseNXtomoMeasurement, EntryData):
    m_def = Section(
        label='ZEISS NXtomo Experimental Record',
        a_eln=dict(lane_width='600px'),
        a_template=dict(measurement_identifiers=dict()),
        a_h5web=H5WebAnnotation(paths=['results/0']),
    )

    instrument_setup = SubSection(section_def=NXtomoInstrumentSetup)
    acquisition_setup = SubSection(section_def=NXtomoAcquisitionSetup)
    results = SubSection(section_def=NXtomoResult, repeats=True)

    def _init_subsections(self):
        if not self.instrument_setup:
            self.instrument_setup = NXtomoInstrumentSetup()
        if not self.acquisition_setup:
            self.acquisition_setup = NXtomoAcquisitionSetup()
        if not self.results:
            self.results = [NXtomoResult()]

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if not self.data_file:
            super().normalize(archive, logger)
            return

        try:
            file_path = archive.m_context.upload_files.raw_file_object(
                self.data_file
            ).os_path
            txrm_data = read_txrm(file_path)

            if 'extraction_error' in txrm_data.metadata:
                logger.warning(
                    f'TXRM Reader Warning: {txrm_data.metadata["extraction_error"]}'
                )

            self._init_subsections()

            self.raw_metadata = txrm_data.metadata
            self.software_version = str(txrm_data.metadata.get('Version', 'Unknown'))

            # Map core numeric setups
            acq_settings = txrm_data.acquisition_settings

            src_voltage = acq_settings.get('SrcVoltage', {})
            if isinstance(src_voltage, dict) and 'float32' in src_voltage:
                self.instrument_setup.source_voltage = src_voltage['float32']

            obj_mag = acq_settings.get('ObjectiveMag', {})
            if isinstance(obj_mag, dict) and 'float32' in obj_mag:
                self.instrument_setup.objective_magnification = obj_mag['float32']

            exp_time = acq_settings.get('ExpTime', {})
            if isinstance(exp_time, dict) and 'float32' in exp_time:
                self.acquisition_setup.exposure_time = exp_time['float32']

            tot_img = acq_settings.get('TotalImages', {})
            if isinstance(tot_img, dict) and 'int32' in tot_img:
                self.acquisition_setup.total_images = tot_img['int32']

            # Map specific TXRM runtime records to results
            res = self.results[0]
            res.total_projections = txrm_data.metadata.get('Total_Projections', 0)
            res.image_data_catalog = txrm_data.image_data_summary
            res.temperature_info = txrm_data.temperature_info
            res.hardware_stability = txrm_data.hw_stability

            # Extract preview image and save as HDF5Dataset
            img_width = 1010
            width_data = txrm_data.image_info.get('ImageWidth', {})
            if isinstance(width_data, dict) and 'int32' in width_data:
                img_width = width_data['int32']

            img_height = 1010
            height_data = txrm_data.image_info.get('ImageHeight', {})
            if isinstance(height_data, dict) and 'int32' in height_data:
                img_height = height_data['int32']

            preview_array = extract_preview_image(
                file_path=file_path, width=img_width, height=img_height
            )

            if preview_array is not None:
                res.preview_image = preview_array

        except Exception as e:
            if logger:
                logger.error(f'Error parsing ZEISS TXRM file: {e}')
            raise e

        super().normalize(archive, logger)


# ==========================================
# 6. ELN ENTRY: ZEISS 3D VOLUME (.txm)
# ==========================================
class ELNZeissTXM(BaseNXtomoMeasurement, EntryData):
    m_def = Section(
        label='ZEISS NXtomo 3D Reconstructed Volume',
        a_eln=dict(lane_width='600px'),
        a_template=dict(measurement_identifiers=dict()),
        a_h5web=H5WebAnnotation(paths=['results/0']),
    )

    instrument_setup = SubSection(section_def=NXtomoInstrumentSetup)
    raw_recon_settings = Quantity(
        type=JSON, description='Complete dictionary of 3D reconstruction parameters.'
    )
    results = SubSection(section_def=NXtomoResult, repeats=True)

    def _init_subsections(self):
        if not self.instrument_setup:
            self.instrument_setup = NXtomoInstrumentSetup()
        if not self.results:
            self.results = [NXtomoResult()]

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if not self.data_file:
            super().normalize(archive, logger)
            return

        try:
            file_path = archive.m_context.upload_files.raw_file_object(
                self.data_file
            ).os_path
            txm_data = read_txm(file_path)

            if 'extraction_error' in txm_data.metadata:
                logger.warning(
                    f'TXM Reader Warning: {txm_data.metadata["extraction_error"]}'
                )

            self._init_subsections()

            self.raw_metadata = txm_data.metadata
            self.software_version = str(txm_data.metadata.get('Version', 'Unknown'))
            self.raw_recon_settings = txm_data.recon_settings

            # Map core numeric setups (TXM stores these in ReconSettings, not AcquisitionSettings)
            recon_settings = txm_data.recon_settings

            src_voltage = recon_settings.get('SourceVoltage', {})
            if isinstance(src_voltage, dict) and 'float32' in src_voltage:
                self.instrument_setup.source_voltage = src_voltage['float32']

            obj_mag = recon_settings.get('LensMagnification', {})
            if isinstance(obj_mag, dict) and 'float32' in obj_mag:
                self.instrument_setup.objective_magnification = obj_mag['float32']

            # Map specific TXM records to results
            res = self.results[0]
            res.total_slices = txm_data.metadata.get('Total_3D_Slices_or_Blocks', 0)
            res.image_data_catalog = txm_data.image_data_summary

            # --- EXTRACT PREVIEW IMAGE FOR TXM ---
            img_width = 1010
            width_data = txm_data.image_info.get('ImageWidth', {})
            if isinstance(width_data, dict) and 'int32' in width_data:
                img_width = width_data['int32']

            img_height = 1010
            height_data = txm_data.image_info.get('ImageHeight', {})
            if isinstance(height_data, dict) and 'int32' in height_data:
                img_height = height_data['int32']

            preview_array = extract_preview_image(
                file_path=file_path, width=img_width, height=img_height
            )

            if preview_array is not None:
                res.preview_image = preview_array

        except Exception as e:
            if logger:
                logger.error(f'Error parsing ZEISS TXM file: {e}')
            raise e

        super().normalize(archive, logger)


# ==========================================
# 7. RAW FILE PLACEHOLDERS
# ==========================================


class RawFileRecipeData(EntryData):
    """Placeholder for the raw RCP file to point to the generated ELN."""

    m_def = Section(label='Raw NXtomo Recipe File')
    measurement = Quantity(
        type=ELNZeissRecipe,
        a_eln=dict(component=ELNComponentEnum.ReferenceEditQuantity),
        description='The editable ELN archive generated from this raw recipe.',
    )


class RawFileTXRMData(EntryData):
    """Placeholder for the raw TXRM file to point to the generated ELN."""

    m_def = Section(label='Raw NXtomo TXRM File')
    measurement = Quantity(
        type=ELNZeissTXRM,
        a_eln=dict(component=ELNComponentEnum.ReferenceEditQuantity),
        description='The editable ELN archive generated from this raw experimental record.',
    )


class RawFileTXMData(EntryData):
    """Placeholder for the raw TXM file to point to the generated ELN."""

    m_def = Section(label='Raw NXtomo TXM File')
    measurement = Quantity(
        type=ELNZeissTXM,
        a_eln=dict(component=ELNComponentEnum.ReferenceEditQuantity),
        description='The editable ELN archive generated from this raw 3D volume.',
    )


m_package.__init_metainfo__()
