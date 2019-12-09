from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
	name='robotic_skin',
	version='0.0.1',
	description='Codes for robotic skin project',
	long_description=readme,
	author='Kandai Watanabe',
	author_email='kandai.watanabe@colorado.edu',
	url='https://github.com/watakandhi/package_template',
	license=license,
	packages=find_packages(exclude=('tests', 'docs')),
	test_suite='tests'
)
