import argparse, logging
import misc.trainHfTokenizer
import misc.snapshotFragPerfDataset
import misc.snapshotFragPerfTraceDataset
import misc.snapshotFragTokenizedShaders
import misc.getTokenizedLengthDistribution
import toyDb.ExperimentDB

logger = logging.getLogger(__name__)

def cli_tokenizer(args):
    if args.train:
        misc.trainHfTokenizer.train(args.train_dest_file)
    else:
        logger.error(f"Nothing to do for cli tokenizer, abort.")

def cli_dataset(args):
    if args.datasetSubCmd == "snapshot":
        misc.snapshotFragPerfDataset.snapshot(
            args.snapshot_train_ratio,
            args.snapshot_dest_file,
            maxTokenizedLength=args.snapshot_max_tokenized_length,
            tokenizerUsed=args.snapshot_tokenizer
        )
    elif args.datasetSubCmd == 'snapshot-tokenized-shader':
        misc.snapshotFragTokenizedShaders.snapshot(
            args.dest_file,
            args.tokenizer
        )
    elif args.datasetSubCmd == 'snapshot-traced':
        misc.snapshotFragPerfTraceDataset.snapshot(
            args.train_ratio,
            args.dest_file,
            None if args.max_tokenized_length == 0 else args.max_tokenized_length
        )
    else:
        logger.error(f"Nothing to do for cli tokenizer, abort.")

def cli_get_tokenized_length_distribution(args):
    misc.getTokenizedLengthDistribution.getTokenizedLengthDistribution(
        tokenizerUsed=args.tokenizer, parallel=args.parallel
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)40s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(prog='ShaderDB')
    parser.add_argument('--database-file', default="")
    subparsers = parser.add_subparsers(dest='command')

    tokenizer_sp = subparsers.add_parser("tokenizer")
    tokenizer_sp.add_argument('--train', action='store_true')
    tokenizer_sp.add_argument('--train-dest-file', default="SpvBpeTokenizer.json")

    dataset_sp = subparsers.add_parser("dataset")
    dataset_sp.add_argument('--tokenizer', type=str, default='spvMultiple')
    dataset_sp.add_argument('--dest-file', default="output.json")

    dataset_subparsers = dataset_sp.add_subparsers(dest="datasetSubCmd")
    dataset_snapshot_sp = dataset_subparsers.add_parser("snapshot")
    dataset_snapshot_sp.add_argument('--train-ratio', type=float, default=0.8)
    dataset_snapshot_sp.add_argument('--max-tokenized-length', type=int, default=4096)

    dataset_snapshot_tokenized_shdr_sp = dataset_subparsers.add_parser('snapshot-tokenized-shader')
    dataset_snapshot_traced_sp = dataset_subparsers.add_parser('snapshot-traced')
    dataset_snapshot_traced_sp.add_argument('--train-ratio', type=float, default=0.8)
    dataset_snapshot_traced_sp.add_argument('--dest-file', type=str)
    dataset_snapshot_traced_sp.add_argument('--max-tokenized-length', type=int)

    tokenized_len_dist_sp = subparsers.add_parser("get-tokenized-length-distribution")
    tokenized_len_dist_sp.add_argument('--tokenizer', default="bpe", help="Tokenizer used")
    tokenized_len_dist_sp.add_argument('--parallel', type=bool, default=True)

    args = parser.parse_args()

    if args.database_file == "":
        toyDb.ExperimentDB.init_from_default_db()
    else:
        toyDb.ExperimentDB.db.init(args.database_file)


    if args.command == 'tokenizer':
        cli_tokenizer(args)
    elif args.command == 'dataset':
        cli_dataset(args)
    elif args.command == 'get-tokenized-length-distribution':
        cli_get_tokenized_length_distribution(args)
    else:
        logger.error(f"Unexpected command {args.command}, abort.")