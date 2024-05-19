import findspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
import argparse
import os

step_files = 1
yellow_schema = StructType(
    [
        StructField("Type", StringType(), nullable=False),
        StructField("VendorID", IntegerType()),
        StructField("Pickup_datetime", TimestampType()),
        StructField("Dropoff_datetime", TimestampType()),
        StructField("Passenger_count", IntegerType()),
        StructField("Trip_distance", FloatType()),
        StructField("Pickup_longitude", FloatType()),
        StructField("Pickup_latitude", FloatType()),
        StructField("Rate_codeID", FloatType()),
        StructField("Store_and_fwd_flag", StringType()),
        StructField("Dropoff_longitude", FloatType()),
        StructField("Dropoff_latitude", FloatType()),
        StructField("Payment_type", IntegerType()),
        StructField("Fare_amount", FloatType()),
        StructField("Extra", FloatType()),
        StructField("Mta_tax", FloatType()),
        StructField("Tip_amount", FloatType()),
        StructField("Tolls_amount", FloatType()),
        StructField("Improvement_surcharge", FloatType()),
        StructField("Total_amount", FloatType()),
    ]
)
green_schema = StructType(
    [
        StructField("Type", StringType(), nullable=False),
        StructField("VendorID", IntegerType()),
        StructField("Pickup_datetime", TimestampType()),
        StructField("Dropoff_datetime", TimestampType()),
        StructField("Store_and_fwd_flag", StringType()),
        StructField("Rate_codeID", IntegerType()),
        StructField("Pickup_longitude", FloatType()),
        StructField("Pickup_latitude", FloatType()),
        StructField("Dropoff_longitude", FloatType()),
        StructField("Dropoff_latitude", FloatType()),
        StructField("Passenger_count", IntegerType()),
        StructField("Trip_distance", FloatType()),
        StructField("Fare_amount", FloatType()),
        StructField("Extra", FloatType()),
        StructField("Mta_tax", FloatType()),
        StructField("Tip_amount", FloatType()),
        StructField("Tolls_amount", FloatType()),
        StructField("Ehail_fee", FloatType()),
        StructField("Improvement_surcharge", FloatType()),
        StructField("Total_amount", FloatType()),
        StructField("Payment_type", IntegerType()),
        StructField("Trip_type", IntegerType()),
    ]
)
full_schema = StructType(
    [StructField(f"_c{i}", StringType(), nullable=True) for i in range(22)]
)


def foreach_batch_function(output_path):
    def batch_func(batch_df, batch_id):
        # Get startHour column
        tmp_df = batch_df.withColumn("startHour", f.hour(f.col("window.start")))

        # For each startHour, write to different directory
        start_hours = tmp_df.select("startHour").distinct().collect()

        if len(start_hours) != 0:
            for start_hour in start_hours:
                h_num = start_hour["startHour"]
                sub_df = tmp_df.where(f.col("startHour") == f.lit(h_num))
                output_dir = f"output-{(h_num + 1) * 60 * 60 * 1000}"

                # sub_df = sub_df.withColumn(
                #     "print", f.concat(f.lit("count:"), f.col("count"))
                # )
                sub_df.select("count").write.mode("overwrite").csv(
                    os.path.join(output_path, output_dir)
                )

    return batch_func


def main(input_path, checkpoint_path, output_path):
    # Creating a SparkSession in Python
    spark = SparkSession.builder.master("local").appName("Spark EX2").getOrCreate()

    # Configures the number of partitions to use when shuffling data for joins or aggregations.
    spark.conf.set("spark.sql.shuffle.partitions", "10")

    # Stream for reading
    full_df = spark.readStream.option("maxFilesPerTrigger", step_files).csv(
        input_path, header=False, schema=full_schema
    )

    yellow_df = full_df.where(full_df["_c0"] == "yellow").selectExpr(
        *[
            f"cast(_c{i} as {field.dataType.simpleString()}) as {field.name}"
            for i, field in enumerate(yellow_schema.fields)
        ]
    )

    green_df = full_df.where(full_df["_c0"] == "green").selectExpr(
        *[
            f"cast(_c{i} as {field.dataType.simpleString()}) as {field.name}"
            for i, field in enumerate(green_schema.fields)
        ]
    )

    unified_df = yellow_df.unionByName(green_df, allowMissingColumns=True)

    by_dropoff = unified_df.groupBy(f.window(f.col("Dropoff_datetime"), "1 hour")).count()

    query = (
        by_dropoff.writeStream.outputMode("update")
        .foreachBatch(foreach_batch_function(output_path))
        .queryName("EventCount")
        .option("checkpointLocation", checkpoint_path)
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spark Job with input, checkpoint, and output arguments"
    )
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument(
        "--checkpoint",
        default="/tmp/checkpoint",
        required=False,
        help="Path to checkpoint data",
    )
    parser.add_argument("--output", required=True, help="Path to output data")

    args = parser.parse_args()

    findspark.init()
    main(args.input, args.checkpoint, args.output)
