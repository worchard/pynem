import setuptools

setuptools.setup(
    name='pynem',
    version='0.0.1',
    description='Python implementation of Nested Effects Models',
    long_description='PyNEM is a package for the learning and handling of Nested Effects Models',
    author='William Orchard',
    author_email='will.r.orchard@gmail.com',
    packages=setuptools.find_packages(exclude=['tests']),
    package_data = {'pynem': ['datasets/toy5a10e2r.pkl']},
    include_package_data=True,
    python_requires='>3.5.0',
    zip_safe=False,
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'numpy'
    ]
)
