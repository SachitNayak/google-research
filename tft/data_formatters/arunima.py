""" Custom formatting functions for Arunima f5 dataset.
    Defines dataset specific column definitions and data transformations.
"""
import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


# Implement formatting functions
class ArunimaFormatter(GenericDataFormatter):
    """Defines and formats data for the arunima f5 dataset.

    This also performs z-score normalization across the entire dataset, hence
    re-uses most of the same functions as volatility.

    Attributes:
    column_definition: Defines input and data type of column used in the
        experiment.
    identifiers: Entity identifiers used in experiments.
    """

    # This defines the types used by each column
    _column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('date', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('second', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('timestamp', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('static_inp', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('n', DataTypes.REAL_VALUED, InputTypes.TARGET),
    ]

    def split_data(self, df, valid_boundary=2015, test_boundary=2017):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
            df: Source data frame to split.
            valid_boundary: Starting year for validation data
            test_boundary: Starting year for test data

        Returns:
            Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')
        index = df['year']
        train = df.loc[index < valid_boundary]
        valid = df.loc[index >= valid_boundary]
        test = df.loc[index >= test_boundary]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
            df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        print(column_definitions)
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # # Format real scalers
        # real_inputs = utils.extract_cols_from_data_type(
        #     DataTypes.REAL_VALUED, column_definitions,
        #     {InputTypes.ID, InputTypes.TIME})

        # data = df[real_inputs].values
        # print("data_shape: ", data.shape)
        # print(data[:10])
        # # vals = data.reshape(-1,1)
        # # print(vals.shape)
        # self._real_scalers = sklearn.preprocessing.StandardScaler().fit()
        # print("done real scalers")
        target_data = df[target_column].values
        old_shape = target_data.shape[0]
        target_data = target_data.reshape(old_shape, 1)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(target_data)  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        num_labels = [str(i) for i in range(1, 41)]
        static_labels = [str(i) for i in range(1, 109)]
        day_labels = [str(i) for i in range(1, 8)]
        date_labels = [str(i) for i in range(1, 32)]
        month_labels = [str(i) for i in range(1, 13)]
        year_labels = [str(i) for i in range(1992, 2021)]
        second_labels = [str(i) for i in [5, 15, 25, 35, 45]]
        for col in categorical_inputs:
            vals = None
            if col == 'day':
                vals = day_labels
            elif col == 'month':
                vals = month_labels
            elif col == 'year':
                vals = year_labels
            elif col == 'date':
                vals = date_labels
            elif col == 'second':
                vals = second_labels
            else:
                vals = static_labels

            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                vals)
            num_classes.append(len(vals))
        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
            df: Data frame to transform.

        Returns:
            Transformed data frame.

        """
        output = df.copy()

        # if self._real_scalers is None and self._cat_scalers is None:
        #     raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        # real_inputs = utils.extract_cols_from_data_type(
        #     DataTypes.REAL_VALUED, column_definitions,
        #     {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        target_col_name = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                             column_definitions)
        target_data = df[target_col_name].values
        old_shape = target_data.shape[0]
        target_data = target_data.reshape(old_shape, 1)
        output[target_col_name] = self._target_scaler.transform(target_data)
        # # Format real inputs
        # output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)
        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
            predictions: Dataframe of model predictions.

        Returns:
            Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                target_data = predictions[col].values
                old_shape = target_data.shape[0]
                target_data = target_data.reshape(old_shape, 1)
                output[col] = self._target_scaler.inverse_transform(target_data)

        return output

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 7*5+1,  # Total width of the Temporal Fusion Decoder
            'num_encoder_steps': 7*5,  # Length of LSTM decoder (ie. # historical inputs)
            'num_epochs': 30,  # Max number of epochs for training
            'early_stopping_patience': 10,  # Early stopping threshold for # iterations with no loss improvement
            'multiprocessing_workers': -1  # Number of multi-processing workers
        }

        return fixed_params

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        return 15000, 8000