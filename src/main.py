"""This file executes the right process."""
import sys
sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

from src.general.argument_parser import create_argument_parser
from src.general.logger_creator import create_logger

from src.optuna_optimization import graph_optimization, classification_head_optimization, threshold_experiment
from src.run_model import run_single_model
from src.data_preparation.create_processed_dataset import process_canary_data, process_litcovid_data
from src.data_preparation.parse_car_xml import parse_document_classification

def main():
    """Main function to start the correct process as specified by the command line."""
    
    # Create and parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create logger with the specified verbosity level
    logger = create_logger(args.verbosity)

    logger.info(f"Verbosity level: {args.verbosity}")
    if args.model_id is not None:
        logger.info(f"Model ID: {args.model_id}")
    else:
        logger.info("Model ID not provided - creating new model.")

    # redirect to correct process
    if args.run_model:
        run_single_model()

    if args.optimize is not None:
        if args.optimize == 'graph':
            graph_optimization()
        elif args.optimize == 'clf_head':
            classification_head_optimization()
        elif args.optimize == 'threshold_experiment':
            if args.pruning_threshold is None:
                assert False, 'Pruning threshold argument not set. Please include --pruning_threshold [float]'
            threshold_experiment(args.pruning_threshold)
        else:
            assert False, f'Optimization method: {args.optimize} not defined.'

    if args.process_data == 'canary':
        process_canary_data()
    elif args.process_data == 'litcovid':
        process_litcovid_data()
    elif args.process_data is not None:
        assert False, f'Specified dataset {args.process_data} not recognised.'

    if args.parse_car_xml:
        parse_document_classification()


if __name__ == "__main__":
    main()
