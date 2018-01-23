from distutils.core import setup

setup(
    name='tensorcv',
    version=0.1,
    packages=['tensorcv'],
    entry_points="""
        [console_scripts]
        tcv=tensorcv:cli
    """
)
