from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, trim


def get_spark():
    """Get or create SparkSession"""
    return SparkSession.builder.appName("FirstETL").getOrCreate()


def run():
    """Main ETL pipeline entry point"""
    spark = get_spark()
    
    # Extract
    df = extract_data(spark)
    
    # Transform
    df_transformed = transform_data(df)
    
    # Load
    load_data(df_transformed)
    
    print("ETL pipeline completed successfully")


def extract_data(spark):
    """Extract data from source"""
    # Example: Read from CSV
    df = spark.read.csv(
        "/Workspace/Users/vinilanair11@gmail.com/etl_repo/data/employee.csv",
        header=True,
        inferSchema=True
    )
    print(f"Extracted {df.count()} records")
    return df


def transform_data(df):
    """Transform the data"""
    # Example transformations
    df_transformed = df.select(
        col("id"),
        upper(trim(col("name"))).alias("name"),
        col("job"),
    ).filter(col("job") == 'IT')
    
    print(f"Transformed to {df_transformed.count()} records")
    return df_transformed


def load_data(df):
    """Load data to destination"""
    # Example: Show the data or write to table
    df.show(10)
    
    # Uncomment to write to Delta table:
    # df.write.format("delta").mode("overwrite").saveAsTable("employee_cleaned")
    
    print("Data loaded successfully")
