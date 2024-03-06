import argparse, logging, sys
from toyDb.experiments import SpvContext
from compete.TracedLinearRegression import TracedLinearRegression
from compete.TracedPerInstLinearRegression import TracedPerInstLinearRegression, TracedPerInstLinearRegressionTorch
from dataset.DatasetBuilder import build_dataset

logger = logging.getLogger(__name__)

def cli_traced_linear_regression(args):
    regressor = TracedLinearRegression(
        args.num_features, args.use_approx_mape, not args.no_trace, args.exclude_first
    )
    trainDataset = build_dataset(args.dataset, "train")
    valDataset = build_dataset(args.dataset, "test")

    if args.load_path != "":
        regressor.load(args.load_path)
    else:
        regressor.train(trainDataset)
    
    print("Train dataset validation:")
    regressor.validate(trainDataset)

    print("val dataset validation:")
    regressor.validate(valDataset)

    if args.save_path != "":
        regressor.save(args.save_path)


def cli_trace_per_inst_linear_regression(args):
    grammar = SpvContext.getDefaultGrammar()

    if args.sklearn:
        regressor = TracedPerInstLinearRegression(
            grammar, args.use_approx_mape
        )
    else:
        regressor = TracedPerInstLinearRegressionTorch(
            grammar, args.use_approx_mape
        )
    trainDataset = build_dataset(args.dataset, "train")
    valDataset = build_dataset(args.dataset, "test")

    if args.load_path != "":
        regressor.load(args.load_path)
    else:
        regressor.train(trainDataset)
    
    print("Train dataset validation:")
    regressor.validate(trainDataset)

    print("val dataset validation:")
    regressor.validate(valDataset)

    if args.save_path != "":
        regressor.save(args.save_path)

def main():
    parser = argparse.ArgumentParser("compete")
    subparsers = parser.add_subparsers(dest='command')
    
    traced_lin_reg_sp = subparsers.add_parser("traced-linear-regression")
    traced_lin_reg_sp.add_argument("--num-features", type=int, default=1)
    traced_lin_reg_sp.add_argument("--no-trace", action='store_true')
    traced_lin_reg_sp.add_argument(
        "--dataset", type=str, default="FragmentPerformanceTracedSnapshotDataset"
    )
    traced_lin_reg_sp.add_argument(
        "--exclude-first", action='store_true'
    )
    traced_lin_reg_sp.add_argument(
        '--use-approx-mape', action='store_true'
    )
    traced_lin_reg_sp.add_argument("--save-path", type=str, default="")
    traced_lin_reg_sp.add_argument("--load-path", type=str, default="")

    traced_per_inst_lin_reg_sp = subparsers.add_parser("traced-per-inst-linear-regression")
    traced_per_inst_lin_reg_sp.add_argument(
        "--dataset", type=str, default="FragmentPerformanceTracedSnapshotDataset"
    )
    traced_per_inst_lin_reg_sp.add_argument(
        '--use-approx-mape', action='store_true'
    )
    traced_per_inst_lin_reg_sp.add_argument(
        '--sklearn', action='store_true'
    )
    traced_per_inst_lin_reg_sp.add_argument("--save-path", type=str, default="")
    traced_per_inst_lin_reg_sp.add_argument("--load-path", type=str, default="")

    args = parser.parse_args()
    if args.command == "traced-linear-regression":
        cli_traced_linear_regression(args)
    elif args.command == "traced-per-inst-linear-regression":
        cli_trace_per_inst_linear_regression(args)

if __name__ == "__main__":
    main()