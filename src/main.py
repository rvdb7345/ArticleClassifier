"""This file executes the right process."""
import sys

from src.data_preparation.create_data_split import create_train_val_test_split
from src.data_preparation.label_embedding.node_to_vec_label_embedding import node2vec_label_embedding
from src.data_preparation.network_generation.create_author_networks import create_author_network
from src.data_preparation.network_generation.create_keyword_networks import create_keyword_network
from src.data_preparation.network_generation.create_label_networks import create_label_network
from src.data_preparation.text_embedding.inference_bert_xml_model import inference_xml_embedder
from src.data_preparation.text_embedding.inference_scibert_model import generate_scibert_embeddings
from src.data_preparation.text_embedding.train_bert_xml_model import train_xml_embedder
from src.data_preparation.text_embedding.train_scibert_model import scibert_finetuning

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

    # do a model run
    if args.run_model:
        run_single_model()

    # optimize the models
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

    # data preparation
    if args.process_data == 'canary':
        process_canary_data()
    elif args.process_data == 'litcovid':
        process_litcovid_data()
    elif args.process_data is not None:
        assert False, f'Specified dataset {args.process_data} not recognised.'

    if args.parse_car_xml:
        parse_document_classification()

    if args.create_data_split:
        create_train_val_test_split()

    if args.generate_network is not None:
        assert len(
            args.generate_network) == 2, f'Provided {len(args.generate_network)} arguments, must be 2 --generate_network [dataset] [network_type]'

        if args.generate_network[1] == 'author':
            create_author_network(args.generate_network[0])
        elif args.generate_network[1] == 'keyword':
            create_keyword_network(args.generate_network[0])
        elif args.generate_network[1] == 'label':
            create_label_network(args.generate_network[0])
        else:
            assert False, f'Network type {args.generate_network[1]} not recognized'

    if args.train_scibert is not None:
        scibert_finetuning(args.train_scibert)
    if args.inference_scibert is not None:
        generate_scibert_embeddings(args.infer_scibert)

    if args.train_xml_embedder is not None:
        train_xml_embedder(args.train_xml_embedder)
    if args.inference_xml_embedder is not None:
        inference_xml_embedder(args.inference_xml_embedder)

    if args.embed_labels is not None:
        node2vec_label_embedding(args.embed_labels)


if __name__ == "__main__":
    main()
