
from optparse import OptionParser
from featurizer import Featurizer
import os

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="run feature extraction on FILE", metavar="FILE")
parser.add_option("-d", "--dir", dest="dir",
                  help="run feature extraction on all files in DIR", metavar="DIR")
parser.add_option("-o", "--out", dest="out",
                  help="output features to OUTDIR", metavar="OUTFILE")
parser.add_option("-v", "--verbose",
                  action="store_false", dest="verbose", default=True,
                  help="don't print verbose messages to stdout")
parser.add_option("-r", "--restart", dest='should_restart',
                    action="store_true" , default=False,
                  help="restart by rewriting output files present in OUTDIR")
def main():
    fearturizer = Featurizer()
    (options, args) = parser.parse_args()
    if not options.filename and not options.dir:
        parser.error('missing -f or -d option')
    if options.filename and options.dir:
        parser.error('please choose only one option to run feature extraction')
    if options.out:
        if not os.path.isdir(options.out):
            parser.error('invalid output path')
        else:
            fearturizer.set_out_dir(options.out)
    if options.filename:
        if not os.path.isfile(options.filename):
            parser.error("" + options.filename + " is not a file")
        else:
            fearturizer.set_in_path(options.filename)
    if options.dir:
        if not os.path.isdir(options.dir):
            parser.error("" + options.dir +" is not a directory")
        else:
            fearturizer.set_in_path(options.dir)
    fearturizer.set_restart(options.should_restart)
    fearturizer.prepare()
    fearturizer.run()

if __name__ == '__main__':
    main()
