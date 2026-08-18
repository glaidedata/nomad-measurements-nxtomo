from nomad.datamodel.context import ServerContext
from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive

# Import NXtomo schema and placeholders
from nomad_measurements_nxtomo.schema_packages.schema_package import (
    ELNZeissRecipe,
    ELNZeissTXM,
    ELNZeissTXRM,
    RawFileRecipeData,
    RawFileTXMData,
    RawFileTXRMData,
)


class NXtomoParser(MatchingParser):
    def is_mainfile(
        self,
        filename: str,
        mime: str,
        buffer: bytes,
        decoded_buffer: str,
        compression: str = None,
    ) -> bool:
        """Gatekeeper for NXtomo OLE2 binary files."""

        filename_lower = filename.lower()

        # Check for both ZEISS formats
        if filename_lower.endswith(('.rcp', '.txrm', '.txm')):
            # These files are binary OLE2 containers, so we ensure the buffer isn't empty.
            if buffer:
                return True

        return False

    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger=None,
        child_archives=None,
    ) -> None:
        logger = logger or archive.m_context.logger

        # Extract the filename, handling server context paths correctly
        data_file = mainfile.rsplit('/', maxsplit=1)[-1]
        if isinstance(archive.m_context, ServerContext):
            data_file = mainfile.split('/raw/', 1)[1]

        filename_lower = data_file.lower()

        # Route to the correct Schema and Placeholder based on the file extension
        if filename_lower.endswith('.rcp'):
            entry = ELNZeissRecipe()
            raw_placeholder_class = RawFileRecipeData
        elif filename_lower.endswith('.txrm'):
            entry = ELNZeissTXRM()
            raw_placeholder_class = RawFileTXRMData
        elif filename_lower.endswith('.txm'):
            entry = ELNZeissTXM()
            raw_placeholder_class = RawFileTXMData
        else:
            logger.error(f'Unsupported NXtomo file format: {data_file}')
            return

        # Assign the file name to the entry
        entry.data_file = data_file

        # Create the separate editable .archive.json file to preserve ELN edits
        archive_name = f'{"".join(data_file.split(".")[:-1])}.archive.json'
        eln_ref = create_archive(entry, archive, archive_name)

        # Link the raw binary file to the generated ELN
        archive.data = raw_placeholder_class(measurement=eln_ref)
