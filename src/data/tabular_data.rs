
//! Helper methods to work with tabular data sets.
use arrayfire::*;
use csv;
use std::fmt;
use std::path::Path;

use super::{DataSet, DataSetError, Scaling, IO};
use crate::errors::*;
use crate::tensor::*;

/// Structure representing tabular data.
pub struct TabularDataSet {
    input_shape: Dim,
    output_shape: Dim,
    num_train_samples: u64,
    num_valid_samples: u64,
    x_train: Tensor,
    y_train: Tensor,
    x_valid: Option<Tensor>,
    y_valid: Option<Tensor>,
    x_test: Option<Tensor>,
    y_test: Option<Tensor>,
    x_train_stats: Option<(Scaling, Tensor, Tensor)>,
    y_train_stats: Option<(Scaling, Tensor, Tensor)>,
}

impl TabularDataSet {

    /// Creates a TabularDataSet from a set of csv files.
    ///
    /// The data are shuffled before being split into training and validation sets.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The path to the csv file containing the input features.
    /// * `outputs` - The path to the csv file containing the output labels.
    /// * `valid_frac` - The fraction of the data used for validation.
    /// * `header` - Flag indicating whether the files have a header.
    pub fn from_csv(inputs: &Path,
                    outputs: &Path,
                    valid_frac: f64,
                    header: bool
    ) -> Result<TabularDataSet, Error> {
        let (in_shape, num_in_samples, in_values) = TabularDataSet::load_data_from_path(&inputs, header)?;
        let (out_shape, num_out_samples, out_values) = TabularDataSet::load_data_from_path(&outputs, header)?;

        if num_in_samples != num_out_samples {
            Err(std::convert::From::from(DataSetError::DimensionMismatch))
        } else {
            let num_samples = num_in_samples;

            let mut x = Tensor::new(&in_values[..], Dim4::new(&[in_shape, 1, 1, num_samples]));
            let mut y = Tensor::new(&out_values[..],  Dim4::new(&[out_shape, 1, 1, num_samples]));

            Tensor::shuffle_mut(&mut x, &mut y);

            // Compute number of samples in training set and validation set
            let num_valid_samples = (valid_frac * num_samples as f64).floor() as u64;
            let num_train_samples = num_samples - num_valid_samples;
            let seqs_train = &[Seq::default(), Seq::default(), Seq::default(), Seq::new(0.0, (num_train_samples - 1) as f64, 1.0)];
            let seqs_valid = &[Seq::default(), Seq::default(), Seq::default(), Seq::new(num_train_samples as f64, (num_samples - 1) as f64, 1.0)];
            let x_train = index(&x, seqs_train);
            let x_valid = index(&x, seqs_valid);
            let y_train = index(&y, seqs_train);
            let y_valid = index(&y, seqs_valid);

            // Create the data set
            Ok(TabularDataSet {
                num_train_samples,
                num_valid_samples,
                input_shape: Dim4::new(&[in_shape, 1, 1, 1]),
                output_shape: Dim4::new(&[out_shape, 1, 1, 1]),
                x_train,
                y_train,
                x_valid: Some(x_valid),
                y_valid: Some(y_valid),
                x_test: None,
                y_test: None,
                x_train_stats: None,
                y_train_stats: None,
            })
        }
    }

    /// Creates a TabularDataSet from Tensors.
    ///
    /// The samples must be stacked along the fourth dimension.
    pub fn from_tensor(x_train: Tensor,
                       y_train: Tensor,
                       x_valid: Option<Tensor>,
                       y_valid: Option<Tensor>,
                       x_test: Option<Tensor>,
                       y_test: Option<Tensor>
    ) -> Result<TabularDataSet, Error> {
        let num_train_samples = x_train.dims()[3];
        let num_valid_samples = match &x_valid {
            Some(x) => x.dims().get()[3],
            None => 0,
        };
        let num_features = x_train.dims()[0];
        let num_outputs = y_train.dims()[0];
        Ok(TabularDataSet {
            num_train_samples,
            num_valid_samples,
            input_shape: Dim4::new(&[num_features, 1, 1, 1]),
            output_shape: Dim4::new(&[num_outputs, 1, 1, 1]),
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
            x_train_stats: None,
            y_train_stats: None,
        })
    }

    /// Loads the content of a csv file into a vector of floats.
    ///
    /// # Return value
    ///
    /// Returns a tuple containing the number of features, the number of samples, and a vector containing the values.
    fn load_data_from_path(path: &Path, header: bool) -> Result<(u64, u64, Vec<PrimitiveType>), DataSetError> {
        //let reader = csv::Reader::from_path(path);
        let reader = csv::ReaderBuilder::new().has_headers(header).from_path(path);
        match reader {
            Ok(mut rdr) => {
                let mut values = Vec::<PrimitiveType>::new();
                let mut input_shape = 0;
                for (i, result) in rdr.records().enumerate() {
                    let record = result.unwrap();
                    if i == 0 {
                        input_shape = record.len() as u64;
                    }
                    for entry in record.iter() {
                        values.push((*entry).parse::<PrimitiveType>().unwrap());
                    }
                }

                let num_samples = values.len() as u64 / input_shape;
                Ok((input_shape, num_samples, values))
            },
            Err(e) => Err(DataSetError::Csv(e))
        }
    }

    /// Normalizes the features of the training, validation, and test (if any) sets.
    ///
    /// The minimum and maximum values of the training features are computed and used to normalize the training,
    /// validation, and test sets. After normalization, the distribution of the features in the training
    /// sets is a uniform distribution within 0 and 1 (assuming that the features originally come from a uniform
    /// distribution).
    pub fn normalize_input(&mut self) {
        self.x_train_stats = Some(self.normalize(IO::Input));
    }

    /// Standardizes the features of the training, validation, and test (if any) sets.
    ///
    /// The mean and standard deviation of the training features are computed and used to standardize the training,
    /// validation, and test sets. After standardization, the distribution of the features in the training set
    /// is a normal distribution with a mean of 0 and standard deviation 1 (assuming that the features originally come
    /// from a Gaussian distribution).
    pub fn standardize_input(&mut self) {
        self.x_train_stats = Some(self.standardize(IO::Input));
    }

    /// Normalizes the labels of the training, validation, and test (if any) sets.
    ///
    /// The minimum and maximum values of the training labels are computed and used to normalize the training,
    /// validation, and test sets. After normalization, the distribution of the labels in the training
    /// sets is a uniform distribution within 0 and 1 (assuming that the labels originally come from a uniform
    /// distribution).
    pub fn normalize_output(&mut self) {

        self.y_train_stats = Some(self.normalize(IO::Output));

        /*
        let y_max = max(&self.y_train, 3);
        let y_min = min(&self.y_train, 3);

        // Normalize y_train, y_valid, and y_test
        self.y_train = div(&sub(&self.y_train, &y_max, true), &sub(&y_max, &y_min, true), true);
        self.y_valid = div(&sub(&self.y_valid, &y_max, true), &sub(&y_max, &y_min, true), true);

        match &mut self.y_test {
            Some(y_test) => {
                self.y_test = Some(div(&sub(y_test, &y_max, true), &sub(&y_max, &y_min, true), true));
            },
            None => (),
        }

        // Save normalization parameters
        self.y_train_stats = Some((Scaling::Normalized, y_min, y_max));
        */
    }


    /// Standardizes the labels of the training, validation, and test (if any) sets.
    ///
    /// The mean and standard deviation of the training labels are computed and used to standardize the training,
    /// validation, and test sets. After standardization, the distribution of the labels in the training set
    /// is a normal distribution with a mean of 0 and standard deviation 1 (assuming that the labels originally come
    /// from a Gaussian distribution).
    pub fn standardize_output(&mut self) {

        self.y_train_stats = Some(self.standardize(IO::Output));

        /*
        let y_mean = mean(&self.y_train, 3);
        let y_std = stdev(&self.y_train, 3);

        // Standardize y_train, y_valid, and y_test
        self.y_train = div(&sub(&self.y_train, &y_mean, true), &y_std, true);
        self.y_valid = div(&sub(&self.y_valid, &y_mean, true), &y_std, true);

        match &mut self.y_test {
            Some(y_test) => {
                self.y_test = Some(div(&sub(y_test, &y_mean, true), &y_std, true));
            },
            None => (),
        }

        // Save standardization parameters
        self.y_train_stats = Some((Scaling::Standardized, y_mean, y_std));
        */
    }

    /// Selects the input or output values.
    fn select_io(&mut self, io: IO) -> (&mut Tensor, Option<&mut Tensor>, Option<&mut Tensor>) {
        match io {
            IO::Input => {
                let test_values = match &mut self.x_test {
                    Some(values) => Some(values),
                    None => None,
                };
                let valid_values = match &mut self.x_valid {
                    Some(values) => Some(values),
                    None => None,
                };
                (&mut self.x_train, valid_values, test_values)
            },
            IO::Output => {
                let test_values = match &mut self.y_test {
                    Some(values) => Some(values),
                    None => None,
                };
                let valid_values = match &mut self.y_valid {
                    Some(values) => Some(values),
                    None => None,
                };
                (&mut self.y_train, valid_values, test_values)
            }
        }
    }

    /// Standardizes the inputs or outputs.
    ///
    /// # Arguments
    ///
    /// * `io` - The IO variant indicating if the inputs or outputs are standardized.
    fn standardize(&mut self, io: IO) -> (Scaling, Tensor, Tensor) {
        let (train_values, valid_values, test_values) = self.select_io(io);

        let mean_value = mean(train_values, 3);
        let standard_deviation = stdev(train_values, 3);

        // Standardize the training, validation, and test sets.
        *train_values = div(&sub(train_values, &mean_value, true), &standard_deviation, true);
        if let Some(valid_values) = valid_values {
            *valid_values = div(&sub(valid_values, &mean_value, true), &standard_deviation, true);
        }

        if let Some(test_values) = test_values {
            *test_values = div(&sub(test_values, &mean_value, true), &standard_deviation, true);
        }

        // Return standardization parameters
        (Scaling::Standardized, mean_value, standard_deviation)
    }

    /// Normalizes the inputs or outputs.
    ///
    /// # Arguments
    ///
    /// * `io` - The IO variant indicating if the inputs or outputs are normalized.
    fn normalize(&mut self, io: IO) -> (Scaling, Tensor, Tensor) {
        let (train_values, valid_values, test_values) = self.select_io(io);

        let max_values = max(train_values, 3);
        let min_values = min(train_values, 3);

        // Normalize y_train, y_valid, and y_test
        *train_values = div(&sub(train_values, &max_values, true), &sub(&max_values, &min_values, true), true);
        if let Some(valid_values) = valid_values {
            *valid_values = div(&sub(valid_values, &max_values, true), &sub(&max_values, &min_values, true), true);
        }
        if let Some(test_values) = test_values {
            *test_values = div(&sub(test_values, &max_values, true), &sub(&max_values, &min_values, true), true);
        }

        // Save normalization parameters
        (Scaling::Normalized, min_values, max_values)
    }
}

impl DataSet for TabularDataSet {
    fn input_shape(&self) -> Dim4 { self.input_shape }

    fn output_shape(&self) -> Dim4 { self.output_shape }

    fn num_train_samples(&self) -> u64 { self.num_train_samples }

    fn num_valid_samples(&self) -> u64 { self.num_valid_samples }

    fn x_train(&self) -> &Tensor {
        &self.x_train
    }

    fn y_train(&self) -> &Tensor {
        &self.y_train
    }

    fn x_valid(&self) -> Option<&Tensor> {
        match &self.x_valid {
            Some(x) => Some(x),
            None => None
        }
    }

    fn y_valid(&self) -> Option<&Tensor> {
        match &self.y_valid {
            Some(y) => Some(y),
            None => None
        }
    }

    fn x_test(&self) -> Option<&Tensor> {
        match &self.x_test {
            Some(values) => Some(values),
            None => None,
        }
    }

    fn y_test(&self) -> Option<&Tensor> {
        match &self.y_test {
            Some(values) => Some(values),
            None => None,
        }
    }

    fn x_train_stats(&self) -> &Option<(Scaling, Tensor, Tensor)> {
        &self.x_train_stats
    }

    fn y_train_stats(&self) -> &Option<(Scaling, Tensor, Tensor)> {
        &self.y_train_stats
    }
}

impl fmt::Display for TabularDataSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "=======")?;
        writeln!(f, "Dataset")?;
        writeln!(f, "=======")?;
        writeln!(f, "Input shape: [{} {} {}]", self.input_shape.get()[0], self.input_shape.get()[1], self.input_shape.get()[2],)?;
        writeln!(f, "Output shape: [{} {} {}]", self.output_shape.get()[0], self.output_shape.get()[1], self.output_shape.get()[2])?;
        writeln!(f, "Number of training samples: {}", self.num_train_samples)?;
        writeln!(f, "Number of validation samples: {}", self.num_valid_samples)?;

        match &self.y_train_stats {
            Some((scaling, c1, c2)) => {
                match scaling {
                    Scaling::Normalized => {
                        writeln!(f, "The output data have been normalized with:")?;
                        af_print!("y_min:", c1);
                        af_print!("y_max:", c2);
                        write!(f, "")?;
                    },
                    Scaling::Standardized => {
                        writeln!(f, "The output data have been standardized with:")?;
                        af_print!("mean:", c1);
                        af_print!("std:", c2);
                        write!(f, "")?;
                    }
                }
            },
            None => write!(f, "")?,
        }
        Ok(())
    }
}