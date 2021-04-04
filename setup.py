from setuptools import setup

setup(
    name='galofeats',
    version='0.1',
    description='Galaxy-halo connection: a feature importances analysis',
    url='https://github.com/Richard-Sti/Particle-Detector',
    author='Richard Stiskalek',
    author_email='richard.stiskalek@protonmail.com',
    license='GPL-3.0',
    packages=['galofeats'],
    install_requires=['scipy',
                      'numpy',
                      'scikit-learn',
                      'joblib',
                      'toml',
                      'matplotlib'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8']
)
