# Before installing camelot, install two dependencies - pip install tk - pip install ghostscript
import camelot

tables = camelot.read_pdf('foo.pdf', page='1')
print(tables)

