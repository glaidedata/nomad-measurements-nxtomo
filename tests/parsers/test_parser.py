import re
from unittest.mock import MagicMock, patch

import pytest
from nomad.datamodel.context import ServerContext
from nomad.datamodel.datamodel import EntryArchive

from nomad_measurements_nxtomo.parsers import parser_entry_point
from nomad_measurements_nxtomo.parsers.parser import NXtomoParser
from nomad_measurements_nxtomo.schema_packages.schema_package import (
    ELNZeissRecipe,
    ELNZeissTXM,
    ELNZeissTXRM,
    RawFileRecipeData,
    RawFileTXMData,
    RawFileTXRMData,
)


def generated_reference(archive_name):
    return f'../upload/archive/{archive_name}#data'


OLE2_SIGNATURE = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'


@pytest.mark.parametrize(
    ('filename', 'matches'),
    [
        ('scan.rcp', True),
        ('scan.RCP', True),
        ('scan.RcP', True),
        ('scan.txrm', True),
        ('scan.TXRM', True),
        ('scan.TxRm', True),
        ('scan.txm', True),
        ('scan.TXM', True),
        ('scan.TxM', True),
        ('scan.txt', False),
    ],
)
def test_nxtomo_entry_point_matches_supported_extensions_case_insensitively(
    filename, matches
):
    """Entry-point matching accepts every suffix case before parser loading."""
    assert bool(re.fullmatch(parser_entry_point.mainfile_name_re, filename)) is matches


@pytest.mark.parametrize('filename', ['scan.rcp', 'scan.txrm', 'scan.txm'])
def test_nxtomo_parser_accepts_supported_ole2_files(filename):
    """All supported extensions require the complete OLE2 signature."""
    assert NXtomoParser().is_mainfile(
        filename=filename,
        mime='application/octet-stream',
        buffer=OLE2_SIGNATURE + b'additional content',
        decoded_buffer='',
    )


@pytest.mark.parametrize(
    ('filename', 'buffer'),
    [
        ('invalid.rcp', b'arbitrary non-empty bytes'),
        ('invalid.txrm', OLE2_SIGNATURE[:4]),
        ('invalid.txm', OLE2_SIGNATURE[:7]),
        ('empty.txrm', b''),
        ('unsupported.wdf', OLE2_SIGNATURE),
    ],
)
def test_nxtomo_parser_rejects_invalid_content(filename, buffer):
    """Invalid content or an unsupported extension cannot match this parser."""
    assert not NXtomoParser().is_mainfile(
        filename=filename,
        mime='application/octet-stream',
        buffer=buffer,
        decoded_buffer='',
    )


@pytest.mark.parametrize(
    ('data_file', 'entry_class', 'placeholder_class'),
    [
        ('scan.rcp', ELNZeissRecipe, RawFileRecipeData),
        ('scan.txrm', ELNZeissTXRM, RawFileTXRMData),
        ('scan.txm', ELNZeissTXM, RawFileTXMData),
    ],
)
@patch('nomad_measurements_nxtomo.parsers.parser.create_archive')
def test_parse_routes_same_stem_files_to_distinct_archives(
    mock_create_archive, data_file, entry_class, placeholder_class
):
    """Each supported format gets its own archive and matching raw placeholder."""
    mock_create_archive.side_effect = lambda _entry, _archive, archive_name: (
        generated_reference(archive_name)
    )
    archive = EntryArchive()
    archive.m_context = MagicMock()

    NXtomoParser().parse(data_file, archive, logger=MagicMock())

    entry, parsed_archive, archive_name = mock_create_archive.call_args.args
    expected_archive_name = f'{data_file}.archive.json'
    assert isinstance(entry, entry_class)
    assert entry.data_file == data_file
    assert parsed_archive is archive
    assert archive_name == expected_archive_name
    assert isinstance(archive.data, placeholder_class)
    assert archive.data.measurement.m_proxy_value == generated_reference(
        expected_archive_name
    )


@pytest.mark.parametrize(
    ('mainfile', 'context', 'expected_data_file'),
    [
        ('sample.v1.txrm', MagicMock(), 'sample.v1.txrm'),
        (
            '/uploads/test/raw/subdir/sample.v1.txrm',
            ServerContext(),
            'subdir/sample.v1.txrm',
        ),
    ],
)
@patch('nomad_measurements_nxtomo.parsers.parser.create_archive')
def test_parse_preserves_dots_and_parent_raw_path(
    mock_create_archive, mainfile, context, expected_data_file
):
    """Archive identities retain internal dots and raw-directory paths."""
    mock_create_archive.side_effect = lambda _entry, _archive, archive_name: (
        generated_reference(archive_name)
    )
    archive = EntryArchive()
    archive.m_context = context

    NXtomoParser().parse(mainfile, archive, logger=MagicMock())

    entry, parsed_archive, archive_name = mock_create_archive.call_args.args
    expected_archive_name = f'{expected_data_file}.archive.json'
    assert entry.data_file == expected_data_file
    assert parsed_archive is archive
    assert archive_name == expected_archive_name
    assert isinstance(archive.data, RawFileTXRMData)
    assert archive.data.measurement.m_proxy_value == generated_reference(
        expected_archive_name
    )


@patch('nomad_measurements_nxtomo.parsers.parser.create_archive')
def test_same_stem_generated_references_do_not_collide(mock_create_archive):
    """Generated references remain tied to each same-stem source archive."""
    mock_create_archive.side_effect = lambda _entry, _archive, archive_name: (
        generated_reference(archive_name)
    )
    parser = NXtomoParser()
    generated_archives = {}

    for data_file in ('scan.rcp', 'scan.txrm', 'scan.txm'):
        archive = EntryArchive()
        archive.m_context = MagicMock()
        parser.parse(data_file, archive, logger=MagicMock())
        archive_name = mock_create_archive.call_args.args[2]
        generated_archives[data_file] = (
            archive_name,
            archive.data.measurement.m_proxy_value,
        )

    assert generated_archives == {
        'scan.rcp': (
            'scan.rcp.archive.json',
            generated_reference('scan.rcp.archive.json'),
        ),
        'scan.txrm': (
            'scan.txrm.archive.json',
            generated_reference('scan.txrm.archive.json'),
        ),
        'scan.txm': (
            'scan.txm.archive.json',
            generated_reference('scan.txm.archive.json'),
        ),
    }
