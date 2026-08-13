from nomad_measurements_nxtomo.parsers.parser import NXtomoParser


def test_nxtomo_parser_is_mainfile():
    """Test that the parser correctly identifies ZEISS NXtomo files."""
    parser = NXtomoParser()

    # 1. Should accept .rcp files with binary content
    assert (
        parser.is_mainfile(
            filename='LFP3_5c.rcp',
            mime='application/octet-stream',
            buffer=b'\xd0\xcf\x11\xe0',  # OLE2 Magic Bytes
            decoded_buffer='',
        )
        is True
    )

    # 2. Should accept .txrm files with binary content
    assert (
        parser.is_mainfile(
            filename='LFP3_5c_4x_3.3um.txrm',
            mime='application/octet-stream',
            buffer=b'\xd0\xcf\x11\xe0',
            decoded_buffer='',
        )
        is True
    )

    # 3. Should reject empty files
    assert (
        parser.is_mainfile(
            filename='empty.txrm',
            mime='application/octet-stream',
            buffer=b'',
            decoded_buffer='',
        )
        is False
    )

    # 4. Should reject unsupported extensions
    assert (
        parser.is_mainfile(
            filename='image.wdf',
            mime='application/octet-stream',
            buffer=b'dummy_data',
            decoded_buffer='',
        )
        is False
    )
