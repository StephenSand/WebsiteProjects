import yfinance as yf
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import date_format
import oci
from oci.object_storage import UploadManager
import os
## pendulum is used for datetime events instead of datetime
import pendulum

from airflow.decorators import dag, task
## To declare a dag here we are using an @dag() decorator
## The @dag means this is a new class inheriting the airflow.decorators.dag class
@dag(
    schedule=None,
    start_date=pendulum.now("UTC"),
    catchup=False,
    tags=["gold_etl"],
)
def gold_etl():
    # Similar to the @dag the classs below uses @task() to inherit the airflow.decorators.task class
    # The name of the function under the @task() is used as the task's unique identifier id
    @task()
    def extract():
        # Retrieving the data from yahoo finance over the past year
        gold_prices = yf.Ticker("GOLD").history("1y")
        return gold_prices
    
    @task()
    def transform(gold_prices):
        # Starting a spark session
        spark = SparkSession.builder.getOrCreate()
        # Resetting index of gold_prices df to transfer all columns to spark
        gold_prices.reset_index(inplace=True)
        # Converting our pandas df to a spark df
        df = spark.createDataFrame(gold_prices)
        # Converting the "Date" column from Timestamp to yyy-MM-dd str format
        df = df.withColumn("Date",  date_format(df.Date, "yyyy-MM-dd"))
        # Making a table view of our df for use with sql queries
        df.createOrReplaceTempView("gold_table")
        # Updating df with Date, Average of High and Low price, Volume columns
        df = spark.sql("SELECT Date, (High + Low) / 2 AS Average_Price, Volume FROM gold_table;")
        # Writing the df to a local CSV file for postgres dbs
        df.coalesce(1).write.csv("pyspark_output/gold_csv", header=True)
        # Writing the df to a local Parquet file for high read calls
        #df.write.parquet("pyspark_output/gold_parquet")
        # Writing the df to a local ORC file for HIVE
        #df.write.orc("pyspark_output/gold_orc")
        # Writing the df to a local Avro file for high write calls
        # For avro you need to clone an external spark-avro package with bash
        #git clone https://github.com/databricks/spark-avro.git
        #cd spark-avro
        #./build/sbt assembly
        #df.write.format("avro").save("gold.avro")
        # Stopping SparkSession
        spark.stop()
        return None
    @task()
    def load():
        # Get list of CSV files in the directory where we saved our df
        paths = []
        for x in os.listdir("pyspark_output/gold_csv"):
            if x.endswith(".csv"): paths.append("pyspark_output/gold_csv/"+x)
        # Establish connection with oci with path to config file
        config = oci.config.from_file("/opt/airflow/.oci/config","DEFAULT")
        # Establish compartment_id, object_storage, and namespace from config file
        compartment_id = config["tenancy"]
        object_storage = oci.object_storage.ObjectStorageClient(config)
        namespace = object_storage.get_namespace().data
        # Specifying bucket_name
        bucket_name = "my-oci-bucket"
        # Base string for new objects placed in the "clean" folder in bucket- later we will add which part it is (0,1,2,etc) and the file extension (.csv)
        object_name = "clean/"+pendulum.now("UTC").to_date_string()+"-part-"
        # Creating the UploadManager to upload files
        upload_manager = UploadManager(object_storage, allow_parallel_uploads=True, parallel_process_count=3)
        # For loop for saving all exported files into object storage bucket with UploadManager while adding the part and file extension to the new object
        i = 0
        for x in paths:
            response = upload_manager.upload_file(namespace, bucket_name, object_name+str(i)+".csv", x)
            i = i+1
        
        # Cleanup files & folders created on drive
        for x in os.listdir("pyspark_output/gold_csv"):
            os.remove("pyspark_output/gold_csv/"+x)
        
        os.rmdir("pyspark_output/gold_csv")
        os.rmdir("pyspark_output")
        return None
    
    ## Now that we have the tasks defined we need to call them in their dependency order
    transform(extract()) >> load()
## Finally now that we have our tasks defined and called within our class,
## and we have our dag defined,
## we call the dag with the dag's unique identifier id "gold_etl()"
## named after the function directly below the @dag decorator
gold_etl()

# the lines below were for local testing
#if __name__ == "__main__":
#    dag.test()
