import logging
import argparse

from databases.ExperimentDb import db, init_from_default_db, init_from_in_memory_db
from databases.ExperimentDb import (
    Environment,
    ImageOnlyShader,
    ImageOnlyResource,
    ImageOnlyExperiment,
    ImageOnlyTrace
)
import cliFuncs.runImageOnly
import cliFuncs.runImageOnlyTrace
import cliFuncs.augmentImageOnly

logger = logging.getLogger(__name__)

def cli_create(db):
    db.create_tables([Environment, ImageOnlyShader, ImageOnlyResource, ImageOnlyExperiment, ImageOnlyTrace])

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)40s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--database-file', type=str, default=None, help="Database file path")
    parser.add_argument(
        '--in-memory-db',
        action='store_true',
        help='This is used for program logic debugging: Create in-memory db and set up the tables'
    )
    parser.add_argument(
        '--log-debug',
        action='store_true',
        help='Toggle log verbosity to DEBUG'
    )

    subparsers = parser.add_subparsers(dest='command')

    create_sp = subparsers.add_parser("create")

    run_sp = subparsers.add_parser("run")
    cliFuncs.runImageOnly.register(run_sp)

    trace_sp = subparsers.add_parser("trace")
    cliFuncs.runImageOnlyTrace.register(trace_sp)

    augment_sp = subparsers.add_parser("augment")
    cliFuncs.augmentImageOnly.register(augment_sp)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.log_debug else logging.INFO,
        format='%(asctime)s - %(name)40s - %(levelname)s - %(message)s'
    )

    if args.in_memory_db:
        assert(args.database_file is None)
        init_from_in_memory_db()
        cli_create(db)
    elif args.database_file is not None:
        db.init(args.database_file)
    else:
        init_from_default_db()

    if args.command == 'create':
        cli_create(db)
    elif args.command == 'run':
        cliFuncs.runImageOnly.cliRun(db, args)
    elif args.command == 'trace':
        cliFuncs.runImageOnlyTrace.cliTrace(db, args)
    elif args.command == 'augment':
        cliFuncs.augmentImageOnly.cliAugment(db, args)
    else:
        logger.error(f"Unexpected command {args.command}, abort.")