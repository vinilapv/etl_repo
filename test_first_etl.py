import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from unittest.mock import patch, MagicMock
import first_etl


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing"""
    spark = SparkSession.builder \
        .appName("TestFirstETL") \
        .master("local[1]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_employee_data(spark):
    """Create sample employee data for testing"""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("job", StringType(), True)
    ])
    
    data = [
        (1, "  john doe  ", "IT"),
        (2, "  jane smith  ", "HR"),
        (3, "  bob wilson  ", "IT"),
        (4, "  alice brown  ", "Finance"),
        (5, "  charlie davis  ", "IT")
    ]
    
    return spark.createDataFrame(data, schema)


class TestGetSpark:
    """Tests for get_spark function"""
    
    def test_get_spark_returns_session(self):
        """Test that get_spark returns a SparkSession"""
        spark = first_etl.get_spark()
        assert spark is not None
        assert isinstance(spark, SparkSession)
    
    def test_get_spark_has_correct_app_name(self):
        """Test that SparkSession has correct app name"""
        spark = first_etl.get_spark()
        assert spark.sparkContext.appName == "FirstETL"


class TestExtractData:
    """Tests for extract_data function"""
    
    @patch('first_etl.SparkSession')
    def test_extract_data_reads_csv(self, mock_spark_class, spark, sample_employee_data):
        """Test that extract_data reads CSV correctly"""
        # Mock the read.csv to return our sample data
        mock_spark = MagicMock()
        mock_spark.read.csv.return_value = sample_employee_data
        
        result = first_etl.extract_data(mock_spark)
        
        # Verify read.csv was called with correct parameters
        mock_spark.read.csv.assert_called_once_with(
            "/Workspace/Users/vinilanair11@gmail.com/etl_repo/data/employee.csv",
            header=True,
            inferSchema=True
        )
        
        assert result is not None
    
    def test_extract_data_returns_dataframe(self, spark, sample_employee_data):
        """Test that extract_data returns a DataFrame"""
        with patch.object(spark.read, 'csv', return_value=sample_employee_data):
            result = first_etl.extract_data(spark)
            assert result is not None
            assert result.count() == 5


class TestTransformData:
    """Tests for transform_data function"""
    
    def test_transform_data_filters_it_jobs(self, sample_employee_data):
        """Test that transform_data filters only IT job records"""
        result = first_etl.transform_data(sample_employee_data)
        
        # Should only have IT employees (3 out of 5)
        assert result.count() == 3
        
        # All records should have job = 'IT'
        jobs = [row.job for row in result.collect()]
        assert all(job == "IT" for job in jobs)
    
    def test_transform_data_uppercases_names(self, sample_employee_data):
        """Test that transform_data converts names to uppercase"""
        result = first_etl.transform_data(sample_employee_data)
        
        names = [row.name for row in result.collect()]
        assert all(name.isupper() for name in names)
    
    def test_transform_data_trims_names(self, sample_employee_data):
        """Test that transform_data trims whitespace from names"""
        result = first_etl.transform_data(sample_employee_data)
        
        names = [row.name for row in result.collect()]
        # Verify no leading/trailing spaces
        assert all(name == name.strip() for name in names)
    
    def test_transform_data_correct_columns(self, sample_employee_data):
        """Test that transform_data returns correct columns"""
        result = first_etl.transform_data(sample_employee_data)
        
        expected_columns = ["id", "name", "job"]
        assert result.columns == expected_columns
    
    def test_transform_data_expected_output(self, sample_employee_data):
        """Test that transform_data produces expected output"""
        result = first_etl.transform_data(sample_employee_data)
        
        result_list = result.collect()
        assert len(result_list) == 3
        
        # Check first record
        assert result_list[0].id == 1
        assert result_list[0].name == "JOHN DOE"
        assert result_list[0].job == "IT"


class TestLoadData:
    """Tests for load_data function"""
    
    def test_load_data_shows_dataframe(self, sample_employee_data, capsys):
        """Test that load_data shows the dataframe"""
        with patch.object(sample_employee_data, 'show') as mock_show:
            first_etl.load_data(sample_employee_data)
            mock_show.assert_called_once_with(10)
    
    def test_load_data_prints_success_message(self, sample_employee_data, capsys):
        """Test that load_data prints success message"""
        with patch.object(sample_employee_data, 'show'):
            first_etl.load_data(sample_employee_data)
            captured = capsys.readouterr()
            assert "Data loaded successfully" in captured.out


class TestRun:
    """Tests for run function (integration test)"""
    
    @patch('first_etl.load_data')
    @patch('first_etl.transform_data')
    @patch('first_etl.extract_data')
    @patch('first_etl.get_spark')
    def test_run_executes_etl_pipeline(self, mock_get_spark, mock_extract, 
                                       mock_transform, mock_load, 
                                       spark, sample_employee_data, capsys):
        """Test that run executes the complete ETL pipeline"""
        # Setup mocks
        mock_get_spark.return_value = spark
        mock_extract.return_value = sample_employee_data
        mock_transform.return_value = sample_employee_data
        
        # Execute
        first_etl.run()
        
        # Verify each step was called
        mock_get_spark.assert_called_once()
        mock_extract.assert_called_once_with(spark)
        mock_transform.assert_called_once()
        mock_load.assert_called_once()
        
        # Verify success message
        captured = capsys.readouterr()
        assert "ETL pipeline completed successfully" in captured.out
    
    @patch('first_etl.load_data')
    @patch('first_etl.transform_data')
    @patch('first_etl.extract_data')
    @patch('first_etl.get_spark')
    def test_run_chains_methods_correctly(self, mock_get_spark, mock_extract,
                                          mock_transform, mock_load,
                                          spark, sample_employee_data):
        """Test that run chains extract -> transform -> load correctly"""
        # Setup mocks to return specific DataFrames
        mock_get_spark.return_value = spark
        mock_extract.return_value = sample_employee_data
        transformed_df = sample_employee_data.limit(3)
        mock_transform.return_value = transformed_df
        
        # Execute
        first_etl.run()
        
        # Verify transform was called with extract's output
        mock_transform.assert_called_once_with(sample_employee_data)
        
        # Verify load was called with transform's output
        mock_load.assert_called_once_with(transformed_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
