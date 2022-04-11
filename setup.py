import setuptools

setuptools.setup(
    name='BS-FDBCK',
    version='1',
    packages=['bs_fdbck', 'bs_fdbck.util', 'bs_fdbck.util.Nd', 'bs_fdbck.util.Nd.sizedist_class_v2',
              'bs_fdbck.util.plot', 'bs_fdbck.util.imports', 'bs_fdbck.util.collocate', 'bs_fdbck.util.EBAS_data',
              'bs_fdbck.util.eusaar_data', 'bs_fdbck.util.slice_average', 'bs_fdbck.util.slice_average.avg_pkg',
              'bs_fdbck.data_info'],
    url='https://github.com/sarambl/BS-FDBCK',
    license='MIT',
    author='Sara Blichner (sarambl)',
    author_email='sara.blichner@aces.su.se',
    description='Analysis code for acp publication'
)
