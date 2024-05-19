import findspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
import argparse
import os

step_files = 1
goldman = [
    [-74.0141012, 40.7152191],
    [-74.013777, 40.7152275],
    [-74.0141027, 40.7138745],
    [-74.0144185, 40.7140753],
]
citigroup = [
    [-74.011869, 40.7217236],
    [-74.009867, 40.721493],
    [-74.010140, 40.720053],
    [-74.012083, 40.720267],
]
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


@f.udf(returnType=BooleanType())
def in_polygon(x, y, polygon):
    num_vertices = len(polygon)
    inside = False
    
    # first point
    p1 = polygon[0]
    
    # For each edge
    for i in range(1, num_vertices + 1):
        # Next point
        p2 = polygon[i % num_vertices]
        
        # if is above the minimum latitude (y)
        if y > min(p1[1], p2[1]):
            # if is below the maximum latitude (y)
            if y <= max(p1[1], p2[1]):
                # if is to the left of the maximum longitude (x)
                if x <= max(p1[0], p2[0]):
                    # get intersection between horizontal line and the edge
                    x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    # if the point is on the same line as the edge or to the left of the x-intersection
                    if p1[0] == p2[0] or x <= x_intersection:
                        # modify flag
                        inside = not inside
                        
        # Store the current point as the first point for the next iteration
        p1 = p2
    # Return final value of the flag
    return inside


def foreach_batch_function(output_path):
    def batch_func(batch_df, batch_id):
        # Get startHour column
        tmp_df = batch_df.withColumn("startHour", f.hour(f.col("window.start")))

        # For each startHour, write to different directory
        start_hours = tmp_df.select("startHour").distinct().collect()

        if len(start_hours) != 0:
            for start_hour in start_hours:
                h_num = start_hour["startHour"]
                sub_df = tmp_df.where(f.col("startHour") == h_num).select(
                    "headquarter", "count"
                )
                final_df = sub_df.withColumn(
                    "print",
                    f.concat(
                        f.lit("("),
                        f.col("headquarter"),
                        f.lit(","),
                        f.col("count").cast("string"),
                        f.lit(")"),
                    ),
                )
                output_dir = f"output-{(h_num + 1) * 60 * 60 * 1000}"
                final_df.select("print").write.mode("overwrite").text(
                    os.path.join(output_path, output_dir)
                )

    return batch_func


def main(input_path, checkpoint_path, output_path):
    # Creating a SparkSession in Python
    spark = SparkSession.builder.master("local").appName("Spark EX3").getOrCreate()

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

    # Create column to mark
    processed_df = unified_df.withColumn(
        "goldman", in_polygon("Dropoff_longitude", "Dropoff_latitude", f.lit(goldman))
    ).withColumn(
        "citigroup",
        in_polygon("Dropoff_longitude", "Dropoff_latitude", f.lit(citigroup)),
    )

    processed_df = processed_df.where("goldman == 1 OR citigroup == 1")
    processed_df = processed_df.withColumn(
        "headquarter", f.when(f.col("goldman") == 1, "goldman").otherwise("citigroup")
    )

    by_dropoff = processed_df.groupBy(
        f.window(f.col("Dropoff_datetime"), "1 hour"), 
        "headquarter"
    ).count()

    query = (
        by_dropoff.writeStream.outputMode("complete")
        .foreachBatch(foreach_batch_function(output_path))
        .queryName("RegionEventCount")
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
