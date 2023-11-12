from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['D:/Medical_Imagery_SRCNN/images/IXI-T1-train'],
                      test_folders=['D:/Medical_Imagery_SRCNN/images/IXI-T1-test'],
                      min_size=100,
                      output_folder='./')
