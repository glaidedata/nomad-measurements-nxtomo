from nomad.config.models.plugins import SchemaPackageEntryPoint

class NXtomoSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_measurements_nxtomo.schema_packages.schema_package import m_package
        return m_package

schema_package_entry_point = NXtomoSchemaPackageEntryPoint(
    name='NXtomo Schema',
    description='Schema package for ZEISS Xradia NXtomo tomographic measurements.',
)