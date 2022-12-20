import click

from peerannot import __version__
from peerannot.runners.run import run
from peerannot.runners.datasets import datasets
from peerannot.runners.train import trainmod
from peerannot.runners.identify import identification

SOURCES = [run, datasets, trainmod, identification]
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(
    name="peerannot",
    cls=click.CommandCollection,
    sources=SOURCES,
    context_settings=CONTEXT_SETTINGS,
    invoke_without_command=True,
)
@click.option("--version", is_flag=True, help="Print version")
@click.pass_context
def peerannot(ctx, version=False):
    if version:
        output = __version__
        output = f"PeerAnnot version {output}"
        click.echo(output)
        raise SystemExit(0)
    if ctx.invoked_subcommand is None:
        print(peerannot.get_help(ctx))


if __name__ == "__main__":
    peerannot()
