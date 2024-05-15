import findspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
import argparse

def main(input_path, checkpoint_path, output_path):
    # define schema for yellow taxi trips
    yellow_taxi_schema = StructType([
        StructField("type", StringType(), nullable=False),
        StructField("VendorID", IntegerType()),
        StructField("tpep_pickup_datetime", TimestampType()),
        StructField("tpep_dropoff_datetime", TimestampType()),
        StructField("passenger_count", IntegerType()),
        StructField("trip_distance", FloatType()),
        StructField("pickup_longitude", FloatType()),
        StructField("pickup_latitude", FloatType()),
        StructField("RatecodeID", FloatType()),
        StructField("store_and_fwd_flag", StringType()),
        StructField("dropoff_longitude", FloatType()),
        StructField("dropoff_latitude", FloatType()),
        StructField("payment_type", IntegerType()),
        StructField("fare_amount", FloatType()),
        StructField("extra", FloatType()),
        StructField("mta_tax", FloatType()),
        StructField("tip_amount", FloatType()),
        StructField("tolls_amount", FloatType()),
        StructField("improvement_surcharge", FloatType()),
        StructField("total_amount", FloatType())
    ])
    # define schema for green taxi trips
    green_taxi_schema = StructType([
        StructField("type", StringType(), nullable=False),
        StructField("VendorID", IntegerType()),
        StructField("lpep_pickup_datetime", TimestampType()),
        StructField("Lpep_dropoff_datetime", TimestampType()),
        StructField("Store_and_fwd_flag", StringType()),
        StructField("RateCodeID", IntegerType()),
        StructField("Pickup_longitude", FloatType()),
        StructField("Pickup_latitude", FloatType()),
        StructField("Dropoff_longitude", FloatType()),
        StructField("Dropoff_latitude", FloatType()),
        StructField("Passenger_count", IntegerType()),
        StructField("Trip_distance", FloatType()),
        StructField("Fare_amount", FloatType()),
        StructField("Extra", FloatType()),
        StructField("MTA_tax", FloatType()),
        StructField("Tip_amount", FloatType()),
        StructField("Tolls_amount", FloatType()),
        StructField("Ehail_fee", FloatType()),
        StructField("improvement_surcharge", FloatType()),
        StructField("Total_amount", FloatType()),
        StructField("Payment_type", IntegerType()),
        StructField("Trip_type", IntegerType())
    ])

    # read the entire data files first
    full_df = spark.readStream\
        .option("header", False)\
        .option("inferSchema", True)\
        .option("maxFilesPerTrigger", 1)\
        .csv(f"hdfs://{input_path}")

    # Filter rows based on the 'type' column
    yellow_df = full_df.where(full_df["_c0"] == "yellow").toDF(schema=yellow_taxi_schema)
    green_df = full_df.where(full_df["_c0"] == "green").toDF(schema=green_taxi_schema)

    # Query
    yellow_by_dropoff = yellow_df\
        .groupBy(f.window(f.col("tpep_dropoff_datetime"), "1 hour"))\
        .count()
    green_by_dropoff = green_df\
        .groupBy(f.window(f.col("Lpep_dropoff_datetime"), "1 hour"))\
        .count()
    
    # Join
    by_dropoff = yellow_by_dropoff.join(green_by_dropoff, on="window")
    
    query = by_dropoff.writeStream\
        .format("console")\
        .outputMode("complete")\
        .queryName("Event_Count")\
        .option("truncate", "false")\
        .start()
    
    query.awaitTermination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Job with input, checkpoint, and output arguments")
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument("--checkpoint", default="/tmp/checkpoint", required=False, help="Path to checkpoint data")
    parser.add_argument("--output", required=True, help="Path to output data")

    args = parser.parse_args()

    findspark.init()

    # Creating a SparkSession in Python
    spark = SparkSession.builder\
        .master("local [*]")\
        .appName("Spark Streaming")\
        .getOrCreate()

    # keep the size of shuffles small
    spark.conf.set("spark.sql.shuffle.partitions", "2")

    main(args.input, args.checkpoint, args.output)