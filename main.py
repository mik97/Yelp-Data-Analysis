import task_pipelines as tp
import tensorflow as tf


def main():
    review_dataset_tasks()


def review_dataset_tasks():
    '''
        Task 1, 2 and 3 that uses the review dataset.
    '''
    # TODO some statistics and visualization

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # tp.task1_pipeline()


if __name__ == "__main__":
    main()
