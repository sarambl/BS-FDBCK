import setuptools

setuptools.setup(
    name='BS-FDBCK',
    version='1',
    packages=['bs_fdbck_clean', 'bs_fdbck_clean.util', 'bs_fdbck_clean.util.Nd', 'bs_fdbck_clean.util.Nd.sizedist_class_v2',
              'bs_fdbck_clean.util.plot', 'bs_fdbck_clean.util.imports', 'bs_fdbck_clean.util.collocate', 'bs_fdbck_clean.util.EBAS_data',
              'bs_fdbck_clean.util.eusaar_data', 'bs_fdbck_clean.util.slice_average', 'bs_fdbck_clean.util.slice_average.avg_pkg',
              'bs_fdbck_clean.data_info'],
    url='https://github.com/sarambl/BS-FDBCK',
    license='MIT',
    author='Sara Blichner (sarambl)',
    author_email='sara.blichner@aces.su.se',
    description='Analysis code for acp publication'
)
