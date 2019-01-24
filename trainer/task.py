import argparse
from . import model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size')
    # Input Arguments
    parser.add_argument('--train_data_files')
    parser.add_argument('--eval_data_files')
    parser.add_argument('--test_data_files')
    # output paths
    parser.add_argument('--model_dir')
    parser.add_argument('--job-dir', default = 'jobdir')

    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by GCP service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)
    
    # Run the training job
    model.train_and_eval(arguments)
