"""This file executes the right process."""
import sys
sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

from src.general.argument_parser import create_argument_parser
from src.general.logger_creator import create_logger

from src.optuna_optimization import graph_optimization, classification_head_optimization
from src.run_model import run_single_model

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
    if args.optimize is not None:
        if args.optimize == 'graph':
            graph_optimization()
        elif args.optimize == 'clf_head':
            classification_head_optimization()
        else:
            assert False, f'Optimization method: {args.optimize} not defined.'
    if args.run_model:
        run_single_model()
        

if __name__ == "__main__":
    main()