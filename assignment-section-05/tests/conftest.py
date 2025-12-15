import logging

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import _load_raw_dataset

np.find_common_type = np.result_type  # very close drop-in replacement

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_input_data():
    data = _load_raw_dataset(file_name=config.app_configs.raw_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data,  # predictors
        data[config.model_configs.target],
        test_size=config.model_configs.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_configs.random_state,
    )

    return X_test
