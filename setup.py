import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='django_query_shortcuts',
    version='1.0.10',
    author='Lewis Munson',
    author_email='73261043+lewis-munson@users.noreply.github.com',
    description='Helpful utilities for reducing repetition in Django queries',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lewis-munson/django-query-shortcuts',
    packages=setuptools.find_packages(),
    install_requires=[
        'django',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
