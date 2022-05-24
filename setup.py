from setuptools import setup

# submodule problem: https://stackoverflow.com/questions/62249207/cant-find-submodule-within-package

setup(
    name='peaktk',
    version='0.1.5',    
    description='A peak day and hour prediction package',
    url='https://github.com/XXXXX/XXXXX',
    author='XXXXX',
    author_email='XXXXXXX',
    license='MIT License',
    packages=['peaktk',
                'peaktk.preprocessing',
                'peaktk.demand_prediction',
                'peaktk.metrics',
                'peaktk.peakday_prediction_monthly',
                'peaktk.peakday_prediction_yearly',
                'peaktk.peakhour_prediction',
                'peaktk.stats',
                'peaktk.dataloader',
                ],
    install_requires=['pandas',
                      'numpy',   
                      'scikit-learn',
                      'matplotlib',
                      'imbalanced-learn>=0.8.0',
                      'statsmodels',
                      'tensorflow==2.5.0'
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',  
        'Topic :: System'
    ],
)
