import click
from robustad.boundary import boundary
from robustad.assembly import assembly
from robustad.fastassembly import  fastassembly
from robustad.lmcc import lmcc
from robustad.plot import  plot
from robustad.diff import diff
from robustad.eval import  eval
from robustad.meanif import  meanif
from robustad.clean import doclean
from robustad.savefig import  savefig
from robustad.util import createdatabase
@click.group()
def cli():
    '''RobusTAD
    nonparametric test detects and quantitates hierarchical topologically associating domains
    '''
    pass



cli.add_command(boundary)
cli.add_command(assembly)
cli.add_command(fastassembly)
cli.add_command(plot)
cli.add_command(diff)
cli.add_command(lmcc)
cli.add_command(eval)
cli.add_command(meanif)
cli.add_command(doclean)
cli.add_command(savefig)

cli.add_command(createdatabase)



if __name__ == '__main__':
    cli()