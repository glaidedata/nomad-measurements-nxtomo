from nomad.config.models.plugins import ParserEntryPoint

class NXtomoParserEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_measurements_nxtomo.parsers.parser import NXtomoParser
        return NXtomoParser(**self.dict())

parser_entry_point = NXtomoParserEntryPoint(
    name='NXtomo Parser',
    description='Parser for ZEISS Xradia NXtomo files (.rcp and .txrm).',
    mainfile_name_re=r'^.*\.(rcp|txrm|RCP|TXRM)$',
)