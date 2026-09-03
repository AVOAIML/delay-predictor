"""maxxflow_cli — the local twins of the Azure ML jobs (plan §11).

Each subcommand is 1:1 with an AML pipeline step. ML logic stays in plain Python
functions inside the libs/modules; this CLI just wires them so ``make`` (local)
and the future AML SDK pipeline (cloud) call the SAME functions unchanged.
"""
